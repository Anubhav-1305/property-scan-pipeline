import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.io_stray import load_capture, quat_to_rot

DATA = Path(__file__).resolve().parents[1] / "data" / "single_room"


def test_rotation_is_orthonormal():
    R = quat_to_rot(0.71, -0.58, -0.07, 0.38)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.skipif(not DATA.exists(), reason="sample data not present")
def test_load_sample():
    c = load_capture(DATA)
    assert c.n_frames > 0
    d = c.load_depth_m(0)
    assert 0.1 < np.median(d[d > 0]) < 10
    assert c.K_depth[0, 0] < c.K_rgb[0, 0]
    assert c.pose(0).shape == (4, 4)
