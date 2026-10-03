"""Door / window detection as gaps in the wall-height point band.

A gap counts as an opening only if there is also evidence of seeing *through*
it (points just behind the wall line). Gaps without that are unscanned wall,
not openings, and are reported under `unverified_gaps`.
"""
import numpy as np


def _runs(mask):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def detect_openings(poly, Pa, floor, ceil, res=0.02):
    top = (ceil - 0.1) if ceil is not None else floor + 2.2
    found, unverified = [], []
    n = len(poly)
    for wi in range(n):
        a, b = poly[wi], poly[(wi + 1) % n]
        ax = 0 if abs(b[0] - a[0]) < abs(b[1] - a[1]) else 1  # const axis
        o = 1 - ax
        lo, hi = sorted([a[o], b[o]])
        L = hi - lo
        d = b - a
        out = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)  # outward (CCW)
        sgn = out[ax] if abs(out[ax]) > 0.5 else 1.0
        rel = (Pa[:, ax] - a[ax]) * np.sign(sgn)  # >0 = outside
        along = Pa[:, o]
        inspan = (along > lo) & (along < hi)
        z = Pa[:, 2] - floor
        nb = max(1, int(round(L / res)))
        edges = np.linspace(lo, hi, nb + 1)

        def occ(zlo, zhi, band, thr):
            m = inspan & (np.abs(rel) < band) & (z > zlo) & (z < min(zhi, top - floor))
            h, _ = np.histogram(along[m], edges)
            return h >= thr

        low = occ(0.15, 0.60, 0.12, 2)
        mid = occ(1.00, 1.90, 0.12, 2)
        k = 5  # close tiny holes (2cm bins)
        ker = np.ones(k)
        low = np.convolve(low, ker, "same") > 0
        mid = np.convolve(mid, ker, "same") > 0
        behind_m = inspan & (rel > 0.2) & (rel < 2.5) & (z > 0.1) & (z < 1.8)
        hb, _ = np.histogram(along[behind_m], edges)

        for kind, gap, w_rng in (("door", ~low & ~mid, (0.55, 2.4)),
                                 ("window", low & ~mid, (0.35, 3.0))):
            for s, e in _runs(gap):
                w = (e - s) * res
                if not (w_rng[0] <= w <= w_rng[1]):
                    continue
                if s * res < 0.1 or (nb - e) * res < 0.1:
                    continue  # touches a corner: occlusion, not an opening
                seen = int(hb[s:e].sum()) >= 30
                p0 = np.zeros(2); p1 = np.zeros(2)
                p0[ax] = p1[ax] = a[ax]
                p0[o], p1[o] = edges[s], edges[e]
                rec = dict(wall_index=wi, type=kind, width_m=float(w),
                           p0=p0.tolist(), p1=p1.tolist())
                (found if seen else unverified).append(rec)
    return found, unverified
