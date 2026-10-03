"""Loader for Stray Scanner style captures.

Layout: rgb.mp4, depth/NNNNNN.png (uint16, millimetres), confidence/NNNNNN.png
(uint8, 0 low / 1 mid / 2 high), odometry.csv (pose per frame), imu.csv,
camera_matrix.csv (intrinsics at RGB resolution).

Depth is lower resolution than RGB (e.g. 256x192 vs 1920x1440), so intrinsics
are rescaled to the depth grid before back-projection.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Capture:
    root: Path
    odometry: pd.DataFrame
    K_rgb: np.ndarray          # 3x3 intrinsics at RGB resolution
    rgb_size: tuple            # (width, height)
    depth_size: tuple          # (width, height)
    n_frames: int

    def depth_path(self, i):
        return self.root / "depth" / f"{int(i):06d}.png"

    def conf_path(self, i):
        return self.root / "confidence" / f"{int(i):06d}.png"

    @property
    def K_depth(self):
        """Intrinsics rescaled to the depth image grid."""
        sx = self.depth_size[0] / self.rgb_size[0]
        sy = self.depth_size[1] / self.rgb_size[1]
        K = self.K_rgb.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return K

    def load_depth_m(self, i):
        """Depth in metres, float32, 0 where invalid."""
        d = cv2.imread(str(self.depth_path(i)), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / 1000.0

    def load_conf(self, i):
        return cv2.imread(str(self.conf_path(i)), cv2.IMREAD_UNCHANGED)

    def pose(self, i):
        """4x4 camera-to-world matrix for frame i."""
        r = self.odometry.iloc[int(i)]
        return pose_matrix(r.x, r.y, r.z, r.qx, r.qy, r.qz, r.qw)


def quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose_matrix(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


def load_capture(root):
    root = Path(root)
    # accept the zip's single top-level folder
    if not (root / "odometry.csv").exists():
        subs = [p for p in root.iterdir() if p.is_dir() and (p / "odometry.csv").exists()]
        if len(subs) != 1:
            raise FileNotFoundError(f"no odometry.csv found under {root}")
        root = subs[0]

    odo = pd.read_csv(root / "odometry.csv", skipinitialspace=True)
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")

    cap = cv2.VideoCapture(str(root / "rgb.mp4"))
    rgb_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    first = cv2.imread(str(root / "depth" / f"{int(odo.frame.iloc[0]):06d}.png"), cv2.IMREAD_UNCHANGED)
    depth_size = (first.shape[1], first.shape[0])

    return Capture(root, odo, K, rgb_size, depth_size, len(odo))


def load_imu(root):
    return pd.read_csv(Path(root) / "imu.csv", skipinitialspace=True)
