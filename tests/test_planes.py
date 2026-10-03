"""Unit tests on SYNTHETIC geometry (a box room with known size).
These test the code only; they are not benchmark data or accuracy claims."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.planes import find_levels, manhattan_angle
from pipeline.lidar_tier import polygon_area, _rectilinearize


def synthetic_room(w=4.0, d=3.0, h=2.6, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    pts.append(np.c_[rng.uniform(0, w, n), rng.uniform(0, d, n), np.zeros(n)])      # floor
    pts.append(np.c_[rng.uniform(0, w, n // 2), rng.uniform(0, d, n // 2), np.full(n // 2, h)])  # ceiling
    for x in (0, w):
        pts.append(np.c_[np.full(n // 4, x), rng.uniform(0, d, n // 4), rng.uniform(0, h, n // 4)])
    for y in (0, d):
        pts.append(np.c_[rng.uniform(0, w, n // 4), np.full(n // 4, y), rng.uniform(0, h, n // 4)])
    return np.vstack(pts)


def test_levels_recover_known_height():
    P = synthetic_room(h=2.6)
    cam_z = np.full(100, 1.4)
    floor, ceil, info = find_levels(P, cam_z)
    assert abs(floor - 0.0) < 0.02
    assert ceil is not None and abs((ceil - floor) - 2.6) < 0.03


def test_no_ceiling_when_not_scanned():
    P = synthetic_room()
    P = P[P[:, 2] < 2.0]  # cut everything above 2 m
    floor, ceil, _ = find_levels(P, np.full(100, 1.4))
    assert ceil is None


def test_manhattan_angle_of_rotated_room():
    P = synthetic_room()
    t = np.deg2rad(23)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    wall = P[(P[:, 2] > 0.4) & (P[:, 2] < 2.0)]
    xy = wall[:, :2] @ R.T
    est = manhattan_angle(xy)
    # recovered angle is the room rotation modulo 90 degrees
    assert abs(((np.rad2deg(est) - 23 + 45) % 90) - 45) < 1.0


def test_rectilinearize_makes_axis_aligned_rectangle():
    noisy = np.array([[0, 0.02], [4.0, -0.03], [4.02, 3.0], [0.01, 2.98]])
    out = _rectilinearize(noisy)
    assert abs(polygon_area(out) - 12.0) < 0.2
    for a, b in zip(out, np.roll(out, -1, axis=0)):
        assert abs(a[0] - b[0]) < 1e-9 or abs(a[1] - b[1]) < 1e-9
