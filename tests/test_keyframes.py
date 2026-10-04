"""Tests the root cause found in the fix loop: dropping blurry frames can leave a
time gap with no overlap. Uses a SYNTHETIC video (noise texture + a blurry stretch)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.video_tier import read_keyframes


def make_video(path, n=240, blur_from=100, blur_to=160):
    rng = np.random.default_rng(0)
    base = (rng.random((240, 320)) * 255).astype(np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(n):
        f = np.roll(base, i, axis=1)
        if blur_from <= i < blur_to:
            f = cv2.GaussianBlur(f, (0, 0), 6)
        w.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    w.release()


def test_sharpest_selection_leaves_no_gap(tmp_path):
    v = tmp_path / "v.mp4"
    make_video(v)
    step = 3  # per_sec=10 at 30 fps
    _, old, _ = read_keyframes(v, per_sec=10, select="threshold")
    _, new, _ = read_keyframes(v, per_sec=10, select="sharpest")
    assert np.diff(old).max() >= 30       # original filter deletes the whole blurry stretch
    assert np.diff(new).max() <= 2 * step  # shipped selection never leaves a gap
