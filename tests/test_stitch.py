"""SYNTHETIC two-room layout (two 4x3 m rooms joined by a 0.9 m doorway).
Tests segmentation/adjacency code only; not benchmark data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import stitch as st


def two_rooms(res=st.RES):
    W, H = int(9.0 / res), int(3.4 / res)
    m = np.zeros((H, W), bool)
    s = lambda a: int(a / res)
    m[s(0.2):s(3.2), s(0.2):s(4.2)] = True          # room A: 4 x 3
    m[s(0.2):s(3.2), s(4.8):s(8.8)] = True          # room B: 4 x 3 (0.6 m wall gap)
    m[s(1.2):s(2.1), s(4.2):s(4.8)] = True          # doorway 0.9 m wide
    return m


def test_two_rooms_found_with_correct_areas():
    lab = st.segment(two_rooms())
    ids = [r for r in np.unique(lab) if r > 0]
    assert len(ids) == 2
    areas = sorted((lab == r).sum() * st.RES ** 2 for r in ids)
    assert all(abs(a - 12.4) < 1.2 for a in areas)  # 12 m2 + share of doorway


def test_adjacency_width_matches_doorway():
    lab = st.segment(two_rooms())
    adj = st.adjacency(lab)
    assert len(adj) == 1
    width = list(adj.values())[0]
    assert abs(width - 0.9) < 0.15


def test_overlap_removal_makes_disjoint():
    a = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float)
    b = np.array([[3.8, 0], [8, 0], [8, 3], [3.8, 3]], float)
    out, before, after = st.remove_overlaps([a, b])
    assert before > 0.5 and after < 1e-6
