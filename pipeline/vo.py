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


def _estimate_poses_before(grays, depths, K, look_back=5, min_inliers=25, bridge="hold"):
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


def _estimate_up_before(points_by_frame, poses, prior=np.array([0.0, -1.0, 0.0]), tol_deg=35,
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


MAX_STEP_M = 1.0
MAX_STEP_DEG = 35.0


def _plausible(T):
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(T[:3, 3]) < MAX_STEP_M and ang < MAX_STEP_DEG


def _estimate_poses_after(grays, depths, K, look_back=6, min_inliers=25, n_try=2, reloc_every=4):
    """Segmented tracking that never trusts a guess.

    * A frame is matched only against recent frames of its own segment; the match
      must be physically plausible (step < 1 m and < 35 deg).
    * If a frame cannot be registered at all (a blind stretch: fast turn, blank
      wall), it starts a NEW segment with its own origin instead of inheriting a
      guessed pose.
    * Frames of a young segment keep trying to relocalise against frames of other
      segments (a revisit). On success the whole segment is moved into the other
      segment's coordinates and merged.
    * At the end only the largest segment is kept; frames in unmerged segments are
      reported as lost (pose None)."""
    feats = [_features(g, 2500) for g in grays]
    n = len(grays)
    poses = [None] * n
    seg = [-1] * n
    inliers = [0] * n
    poses[0], seg[0], inliers[0] = np.eye(4), 0, 999
    members = {0: [0]}
    cur = 0
    recent = [0]  # recent frames of the current segment

    def relocalise(i, avoid_seg):
        cands = [j for s_, m in members.items() if s_ != avoid_seg for j in m]
        if avoid_seg is not None and len(members[avoid_seg]) > look_back + 4:
            cands += members[avoid_seg][:-(look_back + 2)]  # older frames of own segment
        if not cands:
            return None
        pick = [cands[k] for k in np.linspace(0, len(cands) - 1, min(5, len(cands))).astype(int)]
        best = None
        for j in pick:
            T, nn = relative_pose(feats[j], depths[j], feats[i], K, 30)
            if T is not None and (best is None or nn > best[2]):
                best = (np.linalg.inv(T), j, nn)
        return best

    for i in range(1, n):
        best = None
        for j in recent[::-1][:n_try]:
            T, nn = relative_pose(feats[j], depths[j], feats[i], K, min_inliers)
            if T is None or not _plausible(T):
                continue
            if best is None or nn > best[2]:
                best = (np.linalg.inv(T), j, nn)
        if best is None:
            T, nn = relative_pose(feats[recent[-1]], depths[recent[-1]], feats[i], K, 15)
            if T is not None and _plausible(T):
                best = (np.linalg.inv(T), recent[-1], nn)
        if best is not None:
            Tab, j, nn = best
            poses[i], seg[i], inliers[i] = poses[j] @ Tab, cur, nn
            members[cur].append(i)
            recent.append(i)
            recent = recent[-look_back:]
        else:
            rl = relocalise(i, cur)
            if rl is not None and seg[rl[1]] != cur:
                Tab, j, nn = rl
                Pi_other = poses[j] @ Tab  # frame i in the other segment's coordinates
                members[cur].append(i)
                poses[i], seg[i], inliers[i] = np.eye(4), cur, nn
                M = Pi_other @ np.linalg.inv(poses[i])
                tgt = seg[j]
                for k in members[cur]:
                    poses[k] = M @ poses[k]
                    seg[k] = tgt
                members[tgt] += members.pop(cur)
                cur = tgt
                recent = [i]
            else:
                cur = max(members) + 1  # blind stretch: new segment, own origin
                members[cur] = [i]
                poses[i], seg[i], inliers[i] = np.eye(4), cur, 999
                recent = [i]
        # a young segment keeps trying to rejoin the map
        if cur != seg[0] and len(members[cur]) % reloc_every == 0 and len(members) > 1:
            rl = relocalise(i, cur)
            if rl is not None and seg[rl[1]] != cur:
                Tab, j, nn = rl
                Pi_other = poses[j] @ Tab
                M = Pi_other @ np.linalg.inv(poses[i])
                tgt = seg[j]
                for k in members[cur]:
                    poses[k] = M @ poses[k]
                    seg[k] = tgt
                members[tgt] += members.pop(cur)
                cur = tgt
                recent = [i]
    main = max(members, key=lambda s_: len(members[s_]))
    keep = set(members[main])
    for i in range(n):
        if i not in keep:
            poses[i] = None
    held = [p is None for p in poses]
    return poses, inliers, held


def estimate_poses(grays, depths, K, mode="after", **kw):
    """mode='before' reproduces the original tracker (kept for the fix-loop
    before/after run); mode='after' is the shipped fix."""
    if mode == "before":
        return _estimate_poses_before(grays, depths, K, **kw)
    return _estimate_poses_after(grays, depths, K, **kw)


def _estimate_up_after(points_by_frame, poses, prior=np.array([0.0, -1.0, 0.0]), tol_deg=35,
                       iters=600, thresh=0.03, seed=0):
    """Gravity from the largest near-horizontal plane that lies BELOW the camera path
    (a floor is 0.8-2.2 m under a handheld phone; tabletops, ceilings and walls are not).
    Refit on its inliers."""
    rng = np.random.default_rng(seed)
    use = [(p, T) for p, T in zip(points_by_frame, poses) if T is not None]
    P = np.concatenate([(p @ T[:3, :3].T + T[:3, 3]) for p, T in use])
    C = np.array([T[:3, 3] for _, T in use])
    if len(P) > 80000:
        P = P[rng.choice(len(P), 80000, replace=False)]
    cos_tol = np.cos(np.deg2rad(tol_deg))
    cmean = C.mean(0)
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
        h = (cmean - a) @ n  # camera height above this plane
        if not 0.8 < h < 2.2:
            continue
        cnt = int((np.abs((P - a) @ n) < thresh).sum())
        if cnt > best_n:
            best, best_n = (n, a), cnt
    if best is None:
        return prior / np.linalg.norm(prior), 0.0
    n, a = best
    for _ in range(3):  # refit on inliers
        inl = P[np.abs((P - a) @ n) < thresh]
        c = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[-1]
        if n @ prior < 0:
            n = -n
        a = c
    return n, best_n / len(P)


def estimate_up(points_by_frame, poses, mode="after", **kw):
    if mode == "before":
        return _estimate_up_before(points_by_frame, [p for p in poses], **kw)
    return _estimate_up_after(points_by_frame, poses, **kw)


def gravity_rotation(up):
    """Proper rotation R with R @ up = +y."""
    y = up / np.linalg.norm(up)
    x = np.cross(y, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])
