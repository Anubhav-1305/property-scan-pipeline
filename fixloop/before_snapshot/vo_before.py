"""RGB-D visual odometry from predicted depth, and gravity alignment.

Poses are camera-to-world, OpenCV camera axes (x right, y down, z forward), the
same convention as the LiDAR captures. The first camera defines the VO world.
"""
import cv2
import numpy as np


def _intrinsics_scaled(K, sx, sy):
    K = K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def _features(gray, n=3000):
    orb = cv2.ORB_create(nfeatures=n, fastThreshold=12)
    return orb.detectAndCompute(gray, None)


def _backproject(kps, depth, K, dmin=0.3, dmax=6.0):
    pts, idx = [], []
    h, w = depth.shape
    for i, k in enumerate(kps):
        u, v = int(round(k.pt[0])), int(round(k.pt[1]))
        if 0 <= u < w and 0 <= v < h:
            z = depth[v, u]
            if dmin < z < dmax:
                pts.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
                idx.append(i)
    return np.array(pts, np.float64), idx


def relative_pose(feat_a, depth_a, feat_b, K, min_inliers=25):
    """T mapping camera-a coords to camera-b coords, or None."""
    (ka, da), (kb, db) = feat_a, feat_b
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < min_inliers:
        return None, 0
    m = sorted(m, key=lambda x: x.distance)[:800]
    pts3, idx = _backproject([ka[x.queryIdx] for x in m], depth_a, K)
    if len(pts3) < min_inliers:
        return None, 0
    keep = [m[i] for i in idx]
    pts2 = np.array([kb[x.trainIdx].pt for x in keep], np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(pts3, pts2, K, None, iterationsCount=300,
                                             reprojectionError=3.0, confidence=0.995,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < min_inliers:
        return None, 0 if inl is None else len(inl)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec[:, 0]
    return T, len(inl)


def estimate_poses(grays, depths, K, look_back=5, min_inliers=25, bridge="hold"):
    """grays: uint8 images; depths: depth maps resized to image size.
    Returns (poses, inliers, held). Frame 0 is the origin.

    When a frame cannot be matched to any recent tracked frame, it is "held": it
    inherits the previous pose (error bounded by one inter-frame step) so later
    frames can still chain through it (a frame tracked against a held frame
    inherits at most that one-step pose error). Held frames are flagged and
    excluded from fusion. `bridge` selects the gap policy ("hold" = copy the
    previous pose when no match exists)."""
    feats = [_features(g) for g in grays]
    poses, inliers, held = [np.eye(4)], [999], [False]
    kfs = [0]  # keyframes: tracking against them avoids per-frame drift accumulation
    for i in range(1, len(grays)):
        cands = [kfs[-1]] + [i - 1] + kfs[-2:-4:-1] + list(range(i - 2, max(-1, i - look_back - 1), -1))
        seen, done = set(), None
        for thr in (min_inliers, 12):
            for j in cands:
                if j in seen and thr == min_inliers or j < 0:
                    continue
                seen.add(j)
                T, n = relative_pose(feats[j], depths[j], feats[i], K, thr)
                if T is not None:
                    done = (np.linalg.inv(T), j, n)
                    break
            if done:
                break
            seen = set()
        if done is None:
            T, n = relative_pose(feats[i - 1], depths[i - 1], feats[i], K, 8)
            poses.append(poses[i - 1] @ (np.linalg.inv(T) if T is not None else np.eye(4)))
            inliers.append(n if T is not None else 0); held.append(True)
            kfs.append(i)
        else:
            Tab, j, n = done
            poses.append(poses[j] @ Tab)
            inliers.append(n); held.append(False)
            if n < 220 or j != kfs[-1]:
                kfs.append(i)  # tracking against this frame is weakening: new keyframe
    return poses, inliers, held


def estimate_up(points_by_frame, poses, prior=np.array([0.0, -1.0, 0.0]), tol_deg=35,
                iters=400, thresh=0.03, seed=0):
    """Gravity direction in the VO world from the dominant near-horizontal plane
    (floor in practice). `prior` is the first camera's image-up direction."""
    rng = np.random.default_rng(seed)
    P = np.concatenate([(p @ T[:3, :3].T + T[:3, 3]) for p, T in zip(points_by_frame, poses) if T is not None])
    if len(P) > 60000:
        P = P[rng.choice(len(P), 60000, replace=False)]
    cos_tol = np.cos(np.deg2rad(tol_deg))
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ prior < 0:
            n = -n
        if n @ prior < cos_tol:
            continue
        cnt = int((np.abs((P - a) @ n) < thresh).sum())
        if cnt > best_n:
            best, best_n = (n, a), cnt
    if best is None:
        return prior / np.linalg.norm(prior), 0.0
    n, a = best
    inl = P[np.abs((P - a) @ n) < thresh]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1]
    if n @ prior < 0:
        n = -n
    return n, best_n / len(P)


def gravity_rotation(up):
    """Proper rotation R with R @ up = +y."""
    y = up / np.linalg.norm(up)
    x = np.cross(y, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])
