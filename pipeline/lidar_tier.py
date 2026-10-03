"""LiDAR tier: depth + poses -> fused point cloud -> room geometry.

Conventions (verified on the samples): depth is OpenCV style (x right, y down,
z forward) and odometry poses are camera-to-world. World +y is gravity-aligned and
points up; we rotate to a z-up frame (X=x, Y=-z, Z=y). This is a proper rotation
(det=+1), so plans are never mirrored.
"""
import numpy as np

from .io_stray import Capture


def fuse(cap: Capture, stride=5, min_conf=2, dmin=0.3, dmax=4.5, voxel=0.02, frames=None):
    """Return fused points (N,3) in a z-up world frame, metres."""
    K = cap.K_depth
    w, h = cap.depth_size
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    idx = range(0, cap.n_frames, stride) if frames is None else frames
    out = []
    for i in idx:
        d = cap.load_depth_m(i)
        cf = cap.load_conf(i)
        m = (d > dmin) & (d < dmax) & (cf >= min_conf)
        if not m.any():
            continue
        z = d[m]
        x = (u[m] - K[0, 2]) * z / K[0, 0]
        y = (v[m] - K[1, 2]) * z / K[1, 1]
        pc = np.stack([x, y, z], 1)
        T = cap.pose(i)
        out.append((pc @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
        if len(out) >= 40:  # bound memory: downsample in chunks
            out = [voxel_downsample(np.concatenate(out), voxel)]
    P = np.concatenate(out)
    P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)  # z-up, det=+1
    return voxel_downsample(P, voxel)


def voxel_downsample(P, voxel):
    if voxel <= 0:
        return P
    q = np.floor(P / voxel).astype(np.int64)
    _, keep = np.unique(q, axis=0, return_index=True)
    return P[np.sort(keep)]


# ---------------------------------------------------------------- room outline
import cv2

from .planes import find_levels, manhattan_angle, rot2


def _rectilinearize(poly, min_edge=0.25):
    """Snap a closed polygon (N,2) to axis-aligned edges with alternating axes."""
    n = len(poly)
    if n < 4:
        return poly
    edges = []  # (axis, coord, length)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = b - a
        ax = 0 if abs(d[0]) < abs(d[1]) else 1  # axis=0: vertical edge (x const)
        edges.append([ax, (a[ax] + b[ax]) / 2, np.linalg.norm(d)])
    kept = [e for e in edges if e[2] >= min_edge]
    edges = kept if len(kept) >= 4 else edges
    merged = []
    for e in edges:
        if merged and merged[-1][0] == e[0]:
            L = merged[-1][2] + e[2]
            merged[-1][1] = (merged[-1][1] * merged[-1][2] + e[1] * e[2]) / L
            merged[-1][2] = L
        else:
            merged.append(e)
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        a, b = merged[-1], merged.pop(0)
        L = a[2] + b[2]
        a[1] = (a[1] * a[2] + b[1] * b[2]) / L
        a[2] = L
    if len(merged) < 4 or len(merged) % 2:
        return poly
    m = len(merged)
    verts = []
    for i in range(m):
        e0, e1 = merged[i - 1], merged[i]
        v = np.zeros(2)
        v[e0[0]], v[e1[0]] = e0[1], e1[1]
        verts.append(v)
    return np.array(verts)


def room_outline(P, floor, ceil, theta, res=0.05, close_m=0.2, eps_m=0.12):
    """Rectilinear footprint polygon in the Manhattan-aligned frame (metres).
    Returns (polygon (N,2), aligned_xy (M,2) of points used, origin)."""
    top = (ceil - 0.1) if ceil is not None else floor + 2.2
    m = (P[:, 2] > floor - 0.05) & (P[:, 2] < top)
    xy = P[m, :2] @ rot2(-theta).T  # rotate world by -theta
    lo = xy.min(0) - 0.5
    ij = np.floor((xy - lo) / res).astype(int)
    W, H = ij[:, 0].max() + 2, ij[:, 1].max() + 2
    grid = np.zeros((H, W), np.uint8)
    grid[ij[:, 1], ij[:, 0]] = 1
    k = max(3, int(close_m / res) | 1)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # fill holes
    ff = grid.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
    grid = np.where(ff == 2, 0, 1).astype(np.uint8)
    grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid)
    if n < 2:
        raise RuntimeError("no room footprint found")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, eps_m / res, True)[:, 0, :].astype(float)
    poly = ap * res + lo
    poly = _rectilinearize(poly)
    return poly, xy, lo


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def orient_ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    s = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if s > 0 else poly[::-1].copy()


def analyze(P, cam_z):
    """Core geometry for one fused cloud. Returns dict (aligned frame)."""
    floor, ceil, info = find_levels(P, cam_z)
    top = (ceil - 0.2) if ceil is not None else floor + 1.8
    wall = P[(P[:, 2] > floor + 0.4) & (P[:, 2] < top)]
    theta = manhattan_angle(wall[:, :2])
    poly, _, _ = room_outline(P, floor, ceil, theta)
    poly = orient_ccw(poly)
    R = rot2(-theta)
    Pa = np.column_stack([P[:, :2] @ R.T, P[:, 2]])
    return dict(floor=floor, ceil=ceil, info=info, theta=theta, poly=poly, Pa=Pa,
                area=polygon_area(poly),
                height=(ceil - floor) if ceil is not None else None)


def snap_to_walls(poly, Pa, floor, ceil, search=0.4, min_pts=200):
    """Move each footprint edge onto the strongest wall-point line nearby.
    The occupancy outline can overshoot where floor was scanned past a wall;
    wall-height points pin the true wall position."""
    top = (ceil - 0.2) if ceil is not None else floor + 1.8
    W = Pa[(Pa[:, 2] > floor + 0.4) & (Pa[:, 2] < top), :2]
    poly = poly.copy()
    n = len(poly)
    new = poly.copy()
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ax = 0 if abs(b[0] - a[0]) < abs(b[1] - a[1]) else 1  # ax=0: x const
        o = 1 - ax
        lo, hi = sorted([a[o], b[o]])
        sel = W[(W[:, o] > lo) & (W[:, o] < hi) & (np.abs(W[:, ax] - a[ax]) < search)]
        if len(sel) < min_pts:
            continue
        bins = np.arange(a[ax] - search, a[ax] + search + 0.02, 0.02)
        h, e = np.histogram(sel[:, ax], bins)
        k = int(np.argmax(np.convolve(h, np.ones(3), "same")))
        if h[max(0, k - 1):k + 2].sum() < min_pts:
            continue
        c = (e[k] + e[k + 1]) / 2
        near = sel[np.abs(sel[:, ax] - c) < 0.04, ax]
        c = float(np.median(near))
        new[i, ax] = c
        new[(i + 1) % n, ax] = c
    return orient_ccw(_rectilinearize(new, min_edge=0.15))


def wall_evidence(poly, Pa, floor, ceil, bin_m=0.05, tol=0.10):
    """Per-edge fraction of length with wall-height points on the line."""
    top = (ceil - 0.2) if ceil is not None else floor + 1.8
    W = Pa[(Pa[:, 2] > floor + 0.4) & (Pa[:, 2] < top), :2]
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ax = 0 if abs(b[0] - a[0]) < abs(b[1] - a[1]) else 1
        o = 1 - ax
        lo, hi = sorted([a[o], b[o]])
        sel = W[(np.abs(W[:, ax] - a[ax]) < tol) & (W[:, o] > lo) & (W[:, o] < hi)]
        nb = max(1, int(round((hi - lo) / bin_m)))
        h, _ = np.histogram(sel[:, o], bins=nb, range=(lo, hi))
        out.append(float((h >= 5).mean()))
    return out


# ---------------------------------------------------------------- driver
from pathlib import Path

from . import intervals as iv
from .io_stray import load_capture
from .openings import detect_openings
from .schema import empty_result


def _geometry(cap, frames, stride=5):
    P = fuse(cap, stride=stride, frames=frames)
    cam_z = cap.odometry.y.values
    r = analyze(P, cam_z)
    r["poly"] = snap_to_walls(r["poly"], r["Pa"], r["floor"], r["ceil"])
    r["area"] = polygon_area(r["poly"])
    return r


def process_lidar(path, stride=5, n_subsets=3, name=None):
    cap = load_capture(path)
    name = name or Path(path).name
    n = cap.n_frames

    full = _geometry(cap, range(0, n, stride), stride)
    # spread from interleaved subsets of the same capture
    areas, heights = [], []
    for k in range(n_subsets):
        try:
            s = _geometry(cap, range(k * stride, n, stride * n_subsets), stride * n_subsets)
            areas.append(s["area"])
            if s["height"] is not None:
                heights.append(s["height"])
        except Exception:
            pass
    rel_spread = float(np.std(areas) / np.mean(areas)) / 2 if len(areas) >= 2 else 0.02
    h_std = float(np.std(heights)) if len(heights) >= 2 else 0.01

    poly, floor, ceil = full["poly"], full["floor"], full["ceil"]
    ev = wall_evidence(poly, full["Pa"], floor, ceil)
    found, unverified = detect_openings(poly, full["Pa"], floor, ceil)

    # shift to a positive origin for readability
    shift = poly.min(0)
    shp = poly - shift
    walls = []
    for i in range(len(shp)):
        a, b = shp[i], shp[(i + 1) % len(shp)]
        L = float(np.linalg.norm(b - a))
        w = iv.ci(L, iv.length_half(L, rel_spread, ev[i]))
        walls.append(dict(index=i, p0=a.tolist(), p1=b.tolist(), length_m=w,
                          point_evidence=ev[i]))
    openings = []
    for o in found:
        openings.append(dict(wall_index=o["wall_index"], type=o["type"],
                             width_m=iv.ci(o["width_m"], iv.opening_half(o["width_m"])),
                             p0=(np.array(o["p0"]) - shift).tolist(),
                             p1=(np.array(o["p1"]) - shift).tolist()))
    hval = full["height"]
    room = dict(
        id="room_1", name="Room 1",
        polygon_m=shp.tolist(),
        walls=walls,
        floor_area_m2=iv.ci(full["area"], iv.area_half(full["area"], rel_spread)),
        ceiling_height_m=(iv.ci(hval, iv.height_half(h_std)) if hval is not None else
                          {"value": None, "ci95": None, "half_width": None,
                           "note": "ceiling not captured in this scan"}),
        openings=openings,
        unverified_gaps=[{k: (v if k in ("type", "wall_index", "width_m") else None)
                          for k, v in o.items() if k in ("type", "wall_index", "width_m")}
                         for o in unverified],
        damage=[], concealed_damage_flags=[], scope_items=[],
    )
    res = empty_result(name, "lidar")
    res["rooms"].append(room)
    res["meta"] = dict(
        frames=n, stride=stride, floor_z=floor, ceiling_z=ceil,
        camera_height_above_floor=full["info"]["camera_height_above_floor"],
        manhattan_angle_deg=float(np.rad2deg(full["theta"])),
        subset_area_rel_spread=rel_spread,
        intervals="provisional, uncalibrated until benchmark phase",
        drift_correction="none yet (Phase 2)",
    )
    pts = full["Pa"][(full["Pa"][:, 2] > floor + 0.4) & (full["Pa"][:, 2] < floor + 1.8), :2] - shift
    return res, pts


# ------------------------------------------------------------------ top level
import time

from . import intervals as iv
from .openings import detect_openings
from .schema import empty_result


def _geometry(P, cam_z):
    r = analyze(P, cam_z)
    r["poly"] = snap_to_walls(r["poly"], r["Pa"], r["floor"], r["ceil"])
    r["area"] = polygon_area(r["poly"])
    return r


def _extents(poly):
    e = np.ptp(poly, axis=0)
    return float(e[0]), float(e[1])


def process_lidar(cap, name, stride=5, n_sub=3):
    """Full LiDAR-tier run on one capture -> result dict (+ aligned points)."""
    t0 = time.time()
    n = cap.n_frames
    sub_clouds, sub_geo = [], []
    for k in range(n_sub):
        frames = range(k * stride, n, stride * n_sub)
        Pk = fuse(cap, frames=frames)
        sub_clouds.append(Pk)
        try:
            g = _geometry(Pk, cap.odometry.y.values[list(frames)])
            sub_geo.append(g)
        except Exception:
            pass
    P = voxel_downsample(np.concatenate(sub_clouds), 0.02)
    g = _geometry(P, cap.odometry.y.values)

    # spreads across subsets
    def rel_std(vals):
        vals = np.array(vals, float)
        return float(vals.std() / vals.mean()) if len(vals) > 1 and vals.mean() else 0.0

    exts = [_extents(s["poly"]) for s in sub_geo]
    rel_x = rel_std([e[0] for e in exts]) if exts else 0.05
    rel_y = rel_std([e[1] for e in exts]) if exts else 0.05
    rel_a = rel_std([s["area"] for s in sub_geo]) if sub_geo else 0.05
    hs = [s["height"] for s in sub_geo if s["height"] is not None]
    h_std = float(np.std(hs)) if len(hs) > 1 else 0.01

    poly = g["poly"]
    evid = wall_evidence(poly, g["Pa"], g["floor"], g["ceil"])
    found, unverified = detect_openings(poly, g["Pa"], g["floor"], g["ceil"])

    walls = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        L = float(np.linalg.norm(b - a))
        rel = rel_y if abs(b[0] - a[0]) < abs(b[1] - a[1]) else rel_x
        walls.append(dict(index=i, p0=a.tolist(), p1=b.tolist(),
                          length_m=iv.ci(L, iv.length_half(L, rel, evid[i])),
                          point_evidence=evid[i]))
    openings = []
    for o in found:
        w = o["width_m"]
        openings.append(dict(type=o["type"], wall_index=o["wall_index"], p0=o["p0"], p1=o["p1"],
                             width_m=iv.ci(w, iv.opening_half(w))))

    h = g["height"]
    room = dict(
        name="Footprint",
        polygon_m=poly.tolist(),
        walls=walls,
        openings=openings,
        floor_area_m2=iv.ci(g["area"], iv.area_half(g["area"], rel_a)),
        ceiling_height_m=iv.ci(h, iv.height_half(h_std) if h is not None else None),
    )
    res = empty_result(name, "lidar")
    res["rooms"] = [room]
    caveats = []
    if h is None:
        caveats.append("Ceiling not captured in this scan; ceiling height not reported.")
    if min(evid) < 0.5:
        caveats.append("Some wall segments have little point evidence; their lengths carry widened intervals.")
    res["meta"] = dict(
        floor_z=g["floor"], ceiling_z=g["ceil"], camera_height_above_floor=g["info"]["camera_height_above_floor"],
        manhattan_angle_deg=float(np.rad2deg(g["theta"])),
        frames_total=n, frame_stride=stride * n_sub, n_subsets=len(sub_geo),
        unverified_gaps=unverified, caveats=caveats,
        intervals_status="provisional (uncalibrated until benchmark phase)",
        runtime_s=round(time.time() - t0, 1),
    )
    return res, g["Pa"]
