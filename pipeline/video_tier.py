"""Video tier: handheld clip -> predicted depth -> RGB-D odometry -> same geometry
pipeline as the LiDAR tier.

Only rgb.mp4 is read. No LiDAR depth, no ARKit poses, no IMU.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import vo

DEPTH_W = 256
F35_DEFAULT = 26.0  # iPhone 15 main camera, 35mm-equivalent focal length (mm)
DIAG_35 = 43.27


def focal_px(width, height, f35=F35_DEFAULT):
    """Pixel focal length from a 35mm-equivalent focal length (diagonal based)."""
    return f35 * np.hypot(width, height) / DIAG_35


def exif_f35(path):
    try:
        from PIL import Image
        ex = Image.open(path).getexif().get_ifd(0x8769)
        v = ex.get(0xA405)  # FocalLengthIn35mmFilm
        return float(v) if v else None
    except Exception:
        return None


class FramesCapture:
    """Duck-types io_stray.Capture for the geometry pipeline (y-up world)."""

    def __init__(self, depths, poses, K_depth, depth_size, root, info):
        self._d, self._p = depths, poses
        self.K_depth, self.depth_size, self.root = K_depth, depth_size, Path(root)
        self.n_frames = len(depths)
        pos = np.array([T[:3, 3] for T in poses])
        self.odometry = pd.DataFrame(dict(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2]))
        self.info = info

    def load_depth_m(self, i):
        return self._d[int(i)]

    def load_conf(self, i):
        return np.full(self._d[int(i)].shape, 2, np.uint8)

    def pose(self, i):
        return self._p[int(i)]


def _sharp(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


ROT = {None: None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def read_keyframes(video_path, per_sec=12.0, max_frames=450, work_w=960, rotate=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / per_sec)))
    frames, idx = [], []
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                if f.shape[1] > work_w:  # keep memory bounded
                    f = cv2.resize(f, (work_w, int(round(f.shape[0] * work_w / f.shape[1]))), interpolation=cv2.INTER_AREA)
                if ROT[rotate] is not None:
                    f = cv2.rotate(f, ROT[rotate])
                frames.append(f)
                idx.append(i)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video_path}")
    sh = np.array([_sharp(cv2.cvtColor(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)) for f in frames])
    keep = sh >= 0.4 * np.median(sh)
    frames = [f for f, k in zip(frames, keep) if k]
    idx = [i for i, k in zip(idx, keep) if k]
    if len(frames) > max_frames:
        sel = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames, idx = [frames[i] for i in sel], [idx[i] for i in sel]
    return frames, idx, fps


def build_capture(frames, estimator, f35=F35_DEFAULT, root=".", look_back=5, log=print):
    """frames: list of BGR images (full size). Returns FramesCapture (y-up, metric)."""
    h0, w0 = frames[0].shape[:2]
    W = 640
    scale = W / w0
    H = int(round(h0 * scale))
    fpx = focal_px(w0, h0, f35)
    K = np.array([[fpx * scale, 0, W / 2], [0, fpx * scale, H / 2], [0, 0, 1.0]])
    grays, depths = [], []
    for i, f in enumerate(frames):
        small = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        d = estimator.predict(f, index=i)
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        depths.append(d.astype(np.float32))
    poses, inl, held = vo.estimate_poses(grays, depths, K, look_back=look_back)
    ok = [i for i, h in enumerate(held) if not h]
    log(f"visual odometry tracked {len(ok)}/{len(frames)} frames ({sum(held)} bridged)")
    if len(ok) < 3:
        raise RuntimeError("visual odometry failed: too few tracked frames "
                           "(move slower, keep texture in view, avoid blank walls)")

    # depth grid for fusion
    dw, dh = DEPTH_W, int(round(DEPTH_W * H / W))
    K_d = K.copy(); K_d[0, :] *= dw / W; K_d[1, :] *= dh / H
    small_d = [cv2.resize(depths[i], (dw, dh), interpolation=cv2.INTER_AREA) for i in ok]
    # gravity from the dominant horizontal plane
    pts = []
    for dm in small_d:
        v, u = np.mgrid[0:dh:4, 0:dw:4]
        z = dm[v, u]
        m = (z > 0.3) & (z < 6)
        pts.append(np.c_[(u[m] - K_d[0, 2]) * z[m] / K_d[0, 0], (v[m] - K_d[1, 2]) * z[m] / K_d[1, 1], z[m]])
    up, frac = vo.estimate_up(pts, [poses[i] for i in ok])
    Rg = vo.gravity_rotation(up)
    G = np.eye(4); G[:3, :3] = Rg
    wposes = [G @ poses[i] for i in ok]
    info = dict(frames_in=len(frames), frames_tracked=len(ok), frames_bridged=int(sum(held)), mean_inliers=float(np.mean([inl[i] for i in ok[1:]])) if len(ok) > 1 else 0,
                floor_plane_fraction=float(frac), depth_model=estimator.name, f35_mm=f35,
                intrinsics_source="assumed 35mm-equivalent focal length", tracked_index=ok)
    return FramesCapture(small_d, wposes, K_d, (dw, dh), root, info)


def process_video(video_path, name, estimator, rotate=None, **kw):
    from .lidar_tier import process_lidar
    frames, idx, fps = read_keyframes(video_path, rotate=rotate)
    cap = build_capture(frames, estimator, root=Path(video_path).parent)
    res, pts = process_lidar(cap, name, stride=1, n_sub=3, tier="video", prior_rel=0.03,
                             chunk_size=15, **kw)
    res["meta"]["tier_info"] = cap.info
    res["meta"]["caveats"].append(
        "Video tier: depth is predicted, poses are estimated by visual odometry; "
        "interval floor of 3% is an a-priori assumption until calibrated on measured rooms.")
    return res, pts
