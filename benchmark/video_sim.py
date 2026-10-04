"""Stress the video tier with SIMULATED monocular depth error (code-path test).

    python benchmark/video_sim.py data/single_room results/video_sim

Depth here is LiDAR depth plus injected scale error and smooth noise, so this
measures how the video pipeline (visual odometry, gravity alignment, geometry)
responds to depth error. It is a SENSITIVITY test, not video-tier accuracy.
Pseudo-reference = the LiDAR-tier result on the same scan (no ground truth).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.depth_models import SimulatedDepth
from pipeline.io_stray import load_capture
from pipeline.lidar_tier import process_lidar
from pipeline.video_tier import process_video, read_keyframes, build_capture


def umeyama_rigid(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


CONFIGS = [(0.0, 0.0), (0.03, 0.03), (-0.05, 0.03)]


def main(capdir, outroot, which=None):
    capdir = Path(capdir)
    name = capdir.name
    out = Path(outroot) / name
    out.mkdir(parents=True, exist_ok=True)
    stray = load_capture(capdir)
    video = stray.root / "rgb.mp4"
    ref, _ = process_lidar(stray, name)
    ref_area = ref["meta"]["total_footprint_m2"]
    frames, idx, _ = read_keyframes(video, rotate="cw")
    f = out / "video_sim.json"
    rows = json.loads(f.read_text())["rows"] if f.exists() else []
    for ci, (scale_err, sigma) in enumerate(CONFIGS):
        if which is not None and ci not in which:
            continue
        est = SimulatedDepth(stray, idx, scale_err=scale_err, field_sigma=sigma, rotate="cw")
        cap = build_capture(frames, est, f35=28.8, root=capdir, log=lambda *_: None)
        from pipeline.lidar_tier import process_lidar as pl
        res, _ = pl(cap, name, stride=1, n_sub=3, tier="video", prior_rel=0.03)
        tr = np.array(cap.info["tracked_index"])
        gt = stray.odometry[["x", "y", "z"]].values[np.array(idx)[tr]]
        est_p = cap.odometry[["x", "y", "z"]].values
        R, t = umeyama_rigid(est_p, gt)
        ate = float(np.sqrt(((est_p @ R.T + t - gt) ** 2).sum(1).mean()))
        area = res["meta"]["total_footprint_m2"]
        rows.append(dict(depth_scale_error=scale_err, depth_noise=sigma,
                         frames_tracked=f"{cap.info['frames_tracked']}/{cap.info['frames_in']}",
                         ate_rmse_m=round(ate, 3),
                         footprint_m2=round(area, 2),
                         footprint_vs_lidar_pct=round(100 * (area - ref_area) / ref_area, 1),
                         ceiling_m=res["rooms"][0]["ceiling_height_m"]["value"]))
        print(rows[-1], flush=True)
        (out / "video_sim.json").write_text(json.dumps(dict(
        note="SIMULATED depth (LiDAR + injected error). Sensitivity test, not video-tier accuracy. "
             "f35=28.8 mm used so intrinsics match the sample's true focal length.",
        lidar_reference_footprint_m2=ref_area, rows=rows), indent=2))
    return rows


if __name__ == "__main__":
    which = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/video_sim", which)
