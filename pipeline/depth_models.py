"""Monocular depth backends for the video and photo tiers.

Real backend:  DepthAnythingV2Metric  (indoor metric model, runs on CPU).
Test backend:  SimulatedDepth  (LiDAR depth + injected error). TEST ONLY: it lets
us exercise and stress the video/photo code paths in an environment without
model weights. Results produced with it are NOT video/photo-tier accuracy.
"""
import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthEstimator:
    name = "base"

    def predict(self, bgr, index=None):
        """Return float32 depth in metres, any size (resized by the caller)."""
        raise NotImplementedError


class DepthAnythingV2Metric(DepthEstimator):
    name = "depth-anything-v2-metric-indoor-small"

    def __init__(self, model_id=DEFAULT_MODEL, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    def predict(self, bgr, index=None):
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp).predicted_depth
        out = torch.nn.functional.interpolate(out[:, None], size=rgb.shape[:2],
                                              mode="bicubic", align_corners=False)[0, 0]
        return out.float().cpu().numpy()


class SimulatedDepth(DepthEstimator):
    """LiDAR depth from a Stray Scanner capture with injected monocular-style error:
    a global scale error plus a smooth multiplicative error field per frame."""
    name = "SIMULATED (test only)"

    def __init__(self, stray_capture, frame_ids, scale_err=0.0, field_sigma=0.03, seed=0, rotate=None):
        self.cap, self.ids, self.rotate = stray_capture, list(frame_ids), rotate
        self.scale = 1.0 + scale_err
        self.sigma = field_sigma
        self.rng = np.random.default_rng(seed)

    def predict(self, bgr, index=None):
        import cv2
        d = self.cap.load_depth_m(self.ids[index])
        low = self.rng.normal(0, self.sigma, (6, 8)).astype(np.float32)
        field = cv2.resize(low, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_CUBIC)
        out = d * self.scale * (1.0 + field)
        # fill holes so every pixel has a value, as a dense model would
        if (out <= 0).any():
            out[out <= 0] = np.median(out[out > 0]) if (out > 0).any() else 2.0
        out = out.astype(np.float32)
        if self.rotate:
            out = cv2.rotate(out, {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}[self.rotate])
        return out
