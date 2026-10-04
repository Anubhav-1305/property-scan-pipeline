"""Plane-anchored drift correction.

Visual-inertial odometry drifts: heading slowly rotates and positions slide, so
the same wall seen early and late in a long scan lands in two slightly
different places. We anchor the trajectory to the building itself using the
planes that must be shared:

  1. Manhattan heading: walls are axis-aligned in a building, so each chunk of
     the scan is rotated (about its centroid) until its wall points line up with
     the global wall direction.
  2. Wall-plane registration: each chunk is shifted along the two wall axes so
     its wall-plane positions agree with the consensus of all other chunks.
  3. Floor anchoring: each chunk's floor height is shifted to the global floor.

Corrections are rigid per chunk and clipped to small ranges, and every applied
value is logged. This is not a full pose graph; see the report for limits.
"""
import numpy as np

from .planes import find_levels, manhattan_angle, rot2

MAX_YAW_DEG = 8.0
MAX_SHIFT_M = 0.5
BIN = 0.02


def _wall_xy(P, floor, ceil):
    top = (ceil - 0.2) if ceil is not None else floor + 1.8
    m = (P[:, 2] > floor + 0.4) & (P[:, 2] < top)
    return P[m, :2]


def _sharp(a, res=BIN):
    if len(a) < 10:
        return 0.0
    h, _ = np.histogram(a, np.arange(a.min(), a.max() + res, res))
    return float((h.astype(np.float64) ** 2).sum())


def wall_sharpness(P, floor, ceil, theta):
    """Mean points-per-2cm-bin seen by a wall point along the two wall axes.
    Higher = walls pile onto thinner lines = less drift smear."""
    W = _wall_xy(P, floor, ceil)
    if len(W) < 100:
        return 0.0
    a = W @ rot2(-theta).T
    return float((_sharp(a[:, 0]) + _sharp(a[:, 1])) / (2 * len(W)))


def _hist(vals, lo, n):
    h, _ = np.histogram(vals, bins=n, range=(lo, lo + n * BIN))
    return np.convolve(h.astype(float), np.ones(3) / 3, "same")


def correct(chunks, floor, ceil, theta0, iters=2):
    """chunks: list of dicts {P (N,3) z-up world, cam_z (M,)}.
    Returns (corrected list of P arrays, per-chunk log)."""
    Ps = [c["P"].copy() for c in chunks]
    log = [dict(chunk=i, yaw_deg=0.0, dx=0.0, dy=0.0, dz=0.0) for i in range(len(Ps))]

    # 1. heading
    for i, P in enumerate(Ps):
        W = _wall_xy(P, floor, ceil)
        if len(W) < 3000:
            continue
        best, best_s = 0.0, -1.0
        for d in np.arange(-MAX_YAW_DEG, MAX_YAW_DEG + 1e-9, 0.25):
            t = theta0 + np.deg2rad(d)
            a = W @ rot2(-t).T
            s = _sharp(a[:, 0]) + _sharp(a[:, 1])
            if s > best_s:
                best, best_s = d, s
        c = P[:, :2].mean(0)
        R = rot2(-np.deg2rad(best))
        Ps[i][:, :2] = (P[:, :2] - c) @ R.T + c
        log[i]["yaw_deg"] = float(best)

    # 2. wall-plane registration (aligned frame, per axis)
    Rm = rot2(-theta0)
    for _ in range(iters):
        al = [(_wall_xy(P, floor, ceil) @ Rm.T) for P in Ps]
        allw = np.concatenate([a for a in al if len(a)])
        lo = allw.min(0) - 1.0
        n = (np.ptp(allw, 0) / BIN + 100).astype(int)
        glob = [_hist(allw[:, k], lo[k], n[k]) for k in range(2)]
        for i, P in enumerate(Ps):
            if len(al[i]) < 3000:
                continue
            shift = np.zeros(2)
            for k in range(2):
                own = _hist(al[i][:, k], lo[k], n[k])
                ref = glob[k] - own  # consensus of everyone else
                if ref.sum() < 1:
                    continue
                lags = np.arange(-int(MAX_SHIFT_M / BIN), int(MAX_SHIFT_M / BIN) + 1)
                sc = np.array([(np.roll(own, l) * ref).sum() for l in lags])
                base = sc[len(lags) // 2]
                if sc.max() > 1.15 * base:
                    shift[k] = lags[int(np.argmax(sc))] * BIN
            if shift.any():
                tot = np.array([log[i]["dx"], log[i]["dy"]]) + shift
                if np.abs(tot).max() > MAX_SHIFT_M:
                    continue  # would exceed the allowed total; leave as is
                Ps[i][:, :2] += shift @ Rm  # back to world frame
                log[i]["dx"], log[i]["dy"] = float(tot[0]), float(tot[1])

    # 3. floor anchoring
    for i, (P, c) in enumerate(zip(Ps, chunks)):
        try:
            f, _, _ = find_levels(P, c["cam_z"])
        except Exception:
            continue
        dz = floor - f
        if abs(dz) < 0.15:
            Ps[i][:, 2] += dz
            log[i]["dz"] = float(dz)

    # guard: keep the correction only if it makes walls sharper overall
    before = np.concatenate([c["P"] for c in chunks])
    after = np.concatenate(Ps)
    s_before = wall_sharpness(before, floor, ceil, theta0)
    s_after = wall_sharpness(after, floor, ceil, theta0)
    for l in log:
        l["sharp_before"], l["sharp_after"] = s_before, s_after
    if s_after < s_before:
        for l in log:
            l["rejected"] = True
        return [c["P"] for c in chunks], log
    return Ps, log
