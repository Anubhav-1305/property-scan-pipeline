"""Split a whole-property footprint into rooms and compute adjacency.

Method (distance-transform room segmentation): erode the walkable
footprint so narrow necks (doorways, thin corridors) disappear, take the
surviving blobs as room seeds, grow them back by nearest seed, and read
adjacency and passage widths off the shared boundary pixels.

Limit: a corridor wider than ~2*ERODE_M is not separated from its neighbour
and is reported as part of it.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon

from .lidar_tier import _rectilinearize, orient_ccw, polygon_area, snap_to_walls

RES = 0.05
ERODE_M = 0.50
MIN_ROOM_M2 = 1.5


def footprint_mask(Pa, floor, ceil, poly, res=RES):
    """Boolean raster of walkable area, restricted to the footprint polygon."""
    top = (ceil - 0.1) if ceil is not None else floor + 2.2
    m = (Pa[:, 2] > floor - 0.05) & (Pa[:, 2] < top)
    xy = Pa[m, :2]
    lo = np.minimum(xy.min(0), poly.min(0)) - 0.5
    hi = np.maximum(xy.max(0), poly.max(0)) + 0.5
    W, H = np.ceil((hi - lo) / res).astype(int) + 1
    poly_img = np.zeros((H, W), np.uint8)
    pts = np.round((poly - lo) / res).astype(np.int32)
    cv2.fillPoly(poly_img, [pts], 1)
    return poly_img.astype(bool), lo


def segment(mask, res=RES):
    """Label rooms in a boolean mask. Returns int label image (0 = outside)."""
    m8 = mask.astype(np.uint8)
    k = int(ERODE_M / res)
    er = cv2.erode(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n, seeds, stats, _ = cv2.connectedComponentsWithStats(er)
    markers = np.zeros(mask.shape, np.int32)
    nid = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * res * res >= 0.3:
            nid += 1
            markers[seeds == i] = nid
    if nid == 0:
        markers[mask] = 1
        return markers
    # grow seeds back over the footprint by geodesic nearest-seed (breadth-first)
    lab = markers.copy()
    lab[~mask] = 0
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    free = mask & (lab == 0)
    while free.any():
        grown = cv2.dilate(lab.astype(np.float32), cross).astype(np.int32)
        new = free & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
        free &= ~new
    # merge tiny rooms into the neighbour sharing the longest boundary
    changed = True
    while changed:
        changed = False
        for r in [r for r in np.unique(lab) if r > 0]:
            area = (lab == r).sum() * res * res
            if area >= MIN_ROOM_M2:
                continue
            nb = _neighbour_counts(lab, r)
            if nb:
                lab[lab == r] = max(nb, key=nb.get)
                changed = True
                break
    return lab


def _neighbour_counts(lab, r):
    out = {}
    a = lab
    for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        x, y = a[sl_a], a[sl_b]
        for p, q in ((x, y), (y, x)):
            m = (p == r) & (q > 0) & (q != r)
            for v in q[m]:
                out[int(v)] = out.get(int(v), 0) + 1
    return out


def adjacency(lab, res=RES):
    """{(i,j): shared boundary length in metres} for i<j."""
    pairs = {}
    for sl_a, sl_b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                       ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        x, y = lab[sl_a], lab[sl_b]
        m = (x > 0) & (y > 0) & (x != y)
        for a, b in zip(x[m], y[m]):
            key = (int(min(a, b)), int(max(a, b)))
            pairs[key] = pairs.get(key, 0) + 1
    return {k: v * res for k, v in pairs.items()}


def region_polygon(lab, r, lo, res=RES, eps_m=0.12):
    m = (lab == r).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, eps_m / res, True)[:, 0, :].astype(float)
    poly = _rectilinearize(ap * res + lo)
    return orient_ccw(poly)


def remove_overlaps(polys):
    """Make room polygons disjoint. Returns (polys, overlap_before_m2, overlap_after_m2)."""
    sh = [Polygon(p).buffer(0) for p in polys]
    before = sum(sh[i].intersection(sh[j]).area for i in range(len(sh)) for j in range(i + 1, len(sh)))
    for j in range(len(sh)):
        for i in range(j):
            if sh[i].intersection(sh[j]).area > 1e-6:
                d = sh[j].difference(sh[i])
                if d.geom_type == "MultiPolygon":
                    d = max(d.geoms, key=lambda g: g.area)
                sh[j] = d
    after = sum(sh[i].intersection(sh[j]).area for i in range(len(sh)) for j in range(i + 1, len(sh)))
    out = [orient_ccw(np.array(s.exterior.coords)[:-1]) for s in sh]
    return out, float(before), float(after)


def classify(poly, width_limit=1.5, area_limit=6.0):
    """'connector' for small narrow spaces (hallway, passage), else 'room'."""
    sh = Polygon(poly)
    if sh.area < area_limit:
        e = np.ptp(poly, axis=0)
        if min(e) < width_limit:
            return "connector"
    return "room"
