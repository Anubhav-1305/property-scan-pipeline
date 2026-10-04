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


# ------------------------------------------------------------- room placement
def _rot90(v, k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _outward(p0, p1, poly):
    """Outward unit normal of a straight opening on the polygon boundary."""
    d = np.array(p1) - np.array(p0)
    n = np.array([d[1], -d[0]])
    n /= (np.linalg.norm(n) + 1e-12)
    mid = (np.array(p0) + np.array(p1)) / 2
    inside = Polygon(poly).buffer(0).contains(__import__("shapely.geometry", fromlist=["Point"]).Point(*(mid - 0.2 * n)))
    return n if inside else -n


def _transform_room(room, k, shift):
    f = lambda v: (_rot90(np.array(v), k) + shift).tolist()
    r = dict(room)
    r["polygon_m"] = [f(v) for v in room["polygon_m"]]
    r["walls"] = [dict(w, p0=f(w["p0"]), p1=f(w["p1"])) for w in room["walls"]]
    r["openings"] = [dict(o, p0=f(o["p0"]), p1=f(o["p1"])) for o in room["openings"]]
    return r


def place_rooms(results, connections=None):
    """Place independently reconstructed rooms into one plan by matching doorways.

    results: {name: result dict with one room}. connections: list of [nameA, nameB];
    if omitted, rooms are assumed to connect in sorted order (flagged).
    Returns (rooms, adjacency, meta). Rooms with no usable door pair are placed by
    assumption beside their neighbour and flagged placement='assumed'."""
    names = list(results)
    base = {n: results[n]["rooms"][0] for n in names}
    for n in names:
        base[n] = dict(base[n], name=n)
    assumed_conn = connections is None
    if connections is None:
        connections = [[names[i], names[i + 1]] for i in range(len(names) - 1)]
    placed = {names[0]: dict(room=base[names[0]], method="origin")}
    adj, meta = [], dict(connections_source="assumed from folder order" if assumed_conn else "provided", rooms={})
    meta["rooms"][names[0]] = "origin"
    queue = list(connections)
    guard = 0
    while queue and guard < 100:
        guard += 1
        a, b = queue.pop(0)
        if a in placed and b in placed:
            continue
        if a not in placed and b not in placed:
            queue.append([a, b]); continue
        if b in placed:
            a, b = b, a  # a placed, b new
        A, B = placed[a]["room"], base[b]
        doors = lambda R: [o for o in R["openings"] if o["type"] in ("door", "passage")]
        best = None
        for oa in doors(A):
            na = _outward(oa["p0"], oa["p1"], A["polygon_m"])
            ca = (np.array(oa["p0"]) + np.array(oa["p1"])) / 2
            for ob in doors(B):
                nb = _outward(ob["p0"], ob["p1"], B["polygon_m"])
                cb = (np.array(ob["p0"]) + np.array(ob["p1"])) / 2
                for k in range(4):
                    if np.linalg.norm(_rot90(nb, k) + na) > 1e-6:
                        continue  # outward normals must oppose
                    shift = ca - _rot90(cb, k)
                    cand = _transform_room(B, k, shift)
                    ov = sum(Polygon(cand["polygon_m"]).buffer(0).intersection(Polygon(p["room"]["polygon_m"]).buffer(0)).area
                             for p in placed.values())
                    wdiff = abs(oa["width_m"]["value"] - ob["width_m"]["value"])
                    score = ov + wdiff
                    if best is None or score < best[0]:
                        best = (score, cand, oa, ob, ov)
        if best is not None and best[4] < 0.5:
            _, cand, oa, ob, ov = best
            placed[b] = dict(room=cand, method="door-match")
            meta["rooms"][b] = f"door-match to {a} (overlap {ov:.2f} m2)"
            w = (oa["width_m"]["value"] + ob["width_m"]["value"]) / 2
            adj.append(dict(rooms=[a, b], width_m=oa["width_m"], method="door-match",
                            p0=oa["p0"], p1=oa["p1"]))
        else:  # assumed placement: to the right of everything placed so far
            allp = np.vstack([np.array(p["room"]["polygon_m"]) for p in placed.values()])
            shift = np.array([allp[:, 0].max() + 0.5 - np.min(np.array(B["polygon_m"])[:, 0]),
                              allp[:, 1].min() - np.min(np.array(B["polygon_m"])[:, 1])])
            placed[b] = dict(room=_transform_room(B, 0, shift), method="assumed")
            meta["rooms"][b] = f"ASSUMED beside {a} (no matching doorway found)"
            adj.append(dict(rooms=[a, b], method="assumed", width_m=None))
    rooms = []
    for n in names:
        if n in placed:
            r = dict(placed[n]["room"]); r["placement"] = placed[n]["method"]
            rooms.append(r)
    ov_total = sum(Polygon(rooms[i]["polygon_m"]).buffer(0).intersection(Polygon(rooms[j]["polygon_m"]).buffer(0)).area
                   for i in range(len(rooms)) for j in range(i + 1, len(rooms)))
    meta["overlap_m2"] = float(ov_total)
    return rooms, adj, meta
