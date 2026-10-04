"""SYNTHETIC rooms with known doorways: tests the placement code only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.stitch import place_rooms


def mk_room(name, w, d, door_wall):
    poly = [[0, 0], [w, 0], [w, d], [0, d]]  # CCW; wall0 bottom, 1 right, 2 top, 3 left
    walls = [dict(index=i, p0=poly[i], p1=poly[(i + 1) % 4]) for i in range(4)]
    cw = {0: ([w / 2 - .45, 0], [w / 2 + .45, 0]), 1: ([w, d / 2 - .45], [w, d / 2 + .45]),
          2: ([w / 2 - .45, d], [w / 2 + .45, d]), 3: ([0, d / 2 - .45], [0, d / 2 + .45])}[door_wall]
    ci = lambda v: dict(value=v, ci95=[v - .02, v + .02], half_width=.02)
    door = dict(type="door", wall_index=door_wall, p0=cw[0], p1=cw[1], width_m=ci(0.9))
    return dict(rooms=[dict(name=name, polygon_m=poly, walls=walls, openings=[door],
                            floor_area_m2=ci(w * d), ceiling_height_m=ci(2.5))], meta={})


def test_two_rooms_join_without_overlap():
    # room A door on its right wall, room B door on its top wall (needs a 90-degree turn)
    res = {"A": mk_room("A", 4, 3, 1), "B": mk_room("B", 3, 5, 2)}
    rooms, adj, meta = place_rooms(res, [["A", "B"]])
    assert meta["rooms"]["B"].startswith("door-match")
    assert meta["overlap_m2"] < 0.05
    assert adj[0]["method"] == "door-match"


def test_no_door_falls_back_to_flagged_assumption():
    a = mk_room("A", 4, 3, 1)
    b = mk_room("B", 3, 3, 0)
    b["rooms"][0]["openings"] = []
    rooms, adj, meta = place_rooms({"A": a, "B": b}, [["A", "B"]])
    assert [r["placement"] for r in rooms] == ["origin", "assumed"]
    assert meta["overlap_m2"] < 0.05
