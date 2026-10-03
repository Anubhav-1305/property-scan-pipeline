"""Floor/ceiling levels and Manhattan wall orientation."""
import numpy as np


def _hist(z, res=0.01):
    edges = np.arange(z.min(), z.max() + res, res)
    h, e = np.histogram(z, edges)
    hs = np.convolve(h, np.ones(3), "same")
    return hs, (e[:-1] + e[1:]) / 2


def _coverage(xy, cell=0.25):
    return len({(int(a), int(b)) for a, b in np.floor(xy / cell)})


def find_levels(P, cam_z):
    """Return (floor_z, ceiling_z or None, info). Floor must sit well below the
    camera; ceiling must sit above it and form a sharp, populated peak."""
    z = P[:, 2]
    hs, c = _hist(z)
    cz = float(np.median(cam_z))

    fmask = c < cz - 0.6
    fi = int(np.argmax(hs * fmask))
    floor = float(np.median(z[np.abs(z - c[fi]) < 0.02]))

    ceil, info = None, {"ceiling_seen": False}
    above = z[z > cz + 0.3]
    frac_above = len(above) / len(z)
    info["fraction_above_camera"] = float(frac_above)
    if frac_above > 0.05:
        res = 0.05
        edges = np.arange(cz + 0.3, min(above.max(), floor + 4.5) + res, res)
        h, e = np.histogram(above, edges)
        h = np.concatenate([h, np.zeros(6, h.dtype)])  # empty space above the top bin
        e = np.concatenate([e, e[-1] + res * np.arange(1, 7)])
        best, best_r = None, 0.0
        for k in range(len(h) - 6):
            if e[k] < floor + 1.8 or h[k] < 0.03 * len(above):
                continue
            r = h[k] / (h[k + 1:k + 6].mean() + 1.0)  # cliff: dense below, empty above
            if r > best_r:
                best, best_r = k, r
        info["ceiling_cliff_ratio"] = float(best_r)
        if best is not None and best_r >= 8:
            lo, hi = e[best] - res, e[best] + 2 * res
            cand = float(np.median(above[(above > lo) & (above < hi)]))
            # a real ceiling is a plane covering a floor-sized area; a scan that
            # merely stops at some height leaves only a thin ring of wall points
            cov = _coverage(P[(z > lo) & (z < hi), :2]) / max(1, _coverage(P[np.abs(z - floor) < 0.05, :2]))
            info["ceiling_area_coverage"] = float(cov)
            if cov >= 0.35:
                ceil = cand
                info["ceiling_seen"] = True
    info["camera_height_above_floor"] = cz - floor
    return floor, ceil, info


def _sharp(a, res=0.02):
    h, _ = np.histogram(a, np.arange(a.min(), a.max() + res, res))
    return float((h.astype(np.float64) ** 2).sum())


def manhattan_angle(xy, step=0.5, max_pts=200000):
    """Angle (rad, in [0, pi/2)) that makes wall points pile onto axis-aligned lines."""
    if len(xy) > max_pts:
        xy = xy[np.random.default_rng(0).choice(len(xy), max_pts, replace=False)]
    best, best_s = 0.0, -1.0
    for deg in np.arange(0, 90, step):
        t = np.deg2rad(deg)
        c, s = np.cos(t), np.sin(t)
        sc = _sharp(xy @ np.array([c, s])) + _sharp(xy @ np.array([-s, c]))
        if sc > best_s:
            best, best_s = t, sc
    return best


def rot2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])
