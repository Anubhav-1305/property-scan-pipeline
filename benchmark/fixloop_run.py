"""Fix-loop before/after run for the video tier (regenerable).

    python benchmark/fixloop_run.py data/single_room --mode before --depth sim
    python benchmark/fixloop_run.py data/single_room --mode after  --depth sim
    python benchmark/fixloop_run.py data/single_room --mode before --depth model
    python benchmark/fixloop_run.py data/single_room --mode after  --depth model
    python benchmark/fixloop_run.py --table          # build fixloop/before_after.md

--depth sim   : LiDAR depth + injected error (SIMULATED; isolates tracker/gravity).
--depth model : real Depth Anything V2 metric model. Depth maps are cached in
                results/fixloop/depth_cache_<capture>.npz so before and after use
                IDENTICAL depth (deterministic replay). The cache is not committed.
Metrics use the phone's own ARKit poses as a proxy reference for path and gravity
error, and the LiDAR-tier plan as the pseudo-reference for footprint. No tape or
laser ground truth exists for the sample scans.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from pipeline.depth_models import DepthEstimator, SimulatedDepth
from pipeline.io_stray import load_capture
from pipeline.lidar_tier import process_lidar
from pipeline.video_tier import build_capture, read_keyframes

OUT = Path("results/fixloop")
F35 = 28.8  # the sample clip's true focal length; isolates tracking from intrinsics error
ROT_CW = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # sensor camera -> rotated-cw camera


class CachedDepth(DepthEstimator):
    name = "depth-anything-v2-metric-indoor-small (cached)"

    def __init__(self, inner_factory, path):
        self.path, self.inner_factory, self.inner = Path(path), inner_factory, None
        self.store = dict(np.load(self.path)) if self.path.exists() else {}
        self.dirty = False

    def predict(self, bgr, index=None):
        k = str(index)
        if k not in self.store:
            if self.inner is None:
                self.inner = self.inner_factory()
            self.store[k] = cv2.resize(self.inner.predict(bgr, index), (320, 240)).astype(np.float16)
            self.dirty = True
        return self.store[k].astype(np.float32)

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path, **self.store)


def umeyama_rigid(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def run(capdir, mode, depth, per_sec=12.0, max_frames=450, frame_cache=None):
    capdir = Path(capdir)
    name = capdir.name
    stray = load_capture(capdir)
    if frame_cache and Path(frame_cache).exists():
        frames, idx = pickle.load(open(frame_cache, "rb"))
    else:
        frames, idx, _ = read_keyframes(stray.root / "rgb.mp4", per_sec=per_sec, max_frames=max_frames, rotate="cw",
                                         select="threshold" if mode == "before" else "sharpest")
        if frame_cache:
            pickle.dump((frames, idx), open(frame_cache, "wb"))
    if depth == "sim":
        est = SimulatedDepth(stray, idx, scale_err=0.0, field_sigma=0.0, rotate="cw")
    else:
        from pipeline.depth_models import DepthAnythingV2Metric
        est = CachedDepth(DepthAnythingV2Metric, OUT / f"depth_cache_{name}.npz")
    cap = build_capture(frames, est, f35=F35, root=capdir, log=lambda *_: None, mode=mode)
    if hasattr(est, "save"):
        est.save()
    res, _ = process_lidar(cap, name, stride=1, n_sub=3, tier="video", prior_rel=0.03)

    tr = np.array(cap.info["tracked_index"])
    gt_idx = np.array(idx)[tr]
    gt_pos = stray.odometry[["x", "y", "z"]].values[gt_idx]
    est_pos = cap.odometry[["x", "y", "z"]].values
    R, t = umeyama_rigid(est_pos, gt_pos)
    ate = float(np.sqrt(((est_pos @ R.T + t - gt_pos) ** 2).sum(1).mean()))
    path_len = float(np.linalg.norm(np.diff(stray.odometry[["x", "y", "z"]].values[np.array(idx)], axis=0), axis=1).sum())
    # gravity error: estimated world-up as seen by each camera vs ARKit up as seen by that camera
    errs = []
    for k, gi in enumerate(gt_idx):
        up_est = cap.pose(k)[:3, :3].T @ np.array([0, 1.0, 0])
        up_gt = ROT_CW @ (stray.pose(int(gi))[:3, :3].T @ np.array([0, 1.0, 0]))
        errs.append(np.degrees(np.arccos(np.clip(up_est @ up_gt, -1, 1))))
    ref_file = Path("results/samples") / name / "plan.json"
    ref_area = json.loads(ref_file.read_text())["meta"]["total_footprint_m2"] if ref_file.exists() else \
        process_lidar(stray, name)[0]["meta"]["total_footprint_m2"]
    area = res["meta"]["total_footprint_m2"]
    row = dict(capture=name, mode=mode, depth=depth, frames_in=cap.info["frames_in"],
               frames_tracked=cap.info["frames_tracked"], frames_dropped_or_bridged=cap.info["frames_in"] - cap.info["frames_tracked"],
               ate_rmse_m=round(ate, 3), path_length_m=round(path_len, 1),
               gravity_err_deg_median=round(float(np.median(errs)), 1), gravity_err_deg_max=round(float(np.max(errs)), 1),
               footprint_m2=round(float(area), 2), lidar_reference_m2=round(float(ref_area), 2),
               footprint_error_pct=round(100 * (area - ref_area) / ref_area, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{depth}_{mode}_{name}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    return row


def table():
    rows = sorted(OUT.glob("*_*_*.json"))
    data = [json.loads(p.read_text()) for p in rows]
    lines = ["# Fix loop: before vs after (video tier)", "",
             "| depth | mode | tracked | path error (m) | gravity err median/max (deg) | footprint (m2) | vs LiDAR ref |",
             "|---|---|---|---|---|---|---|"]
    for d in sorted(data, key=lambda r: (r["depth"], r["mode"] != "before")):
        lines.append(f"| {d['depth']} | {d['mode']} | {d['frames_tracked']}/{d['frames_in']} | {d['ate_rmse_m']} | "
                     f"{d['gravity_err_deg_median']} / {d['gravity_err_deg_max']} | {d['footprint_m2']} | {d['footprint_error_pct']}% |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?")
    ap.add_argument("--mode", choices=["before", "after"])
    ap.add_argument("--depth", choices=["sim", "model"], default="sim")
    ap.add_argument("--frame-cache")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        print(table())
    else:
        run(a.capture, a.mode, a.depth, frame_cache=a.frame_cache)
