"""Photo tier: per-room folders of 2-8 stills -> one stitched whole-property plan.

Each folder is one room. Stills are treated as a sparse sequence: predicted depth
per photo, relative poses from RGB-D matching between photos (so neighbouring
photos must overlap), then the same geometry pipeline as the other tiers.
Rooms are joined by matching doorways (see stitch.place_rooms). Intervals carry
an 8% a-priori floor, to be calibrated on measured rooms.
"""
from pathlib import Path

import cv2
import numpy as np

from .video_tier import build_capture, exif_f35, F35_DEFAULT

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic"}


def list_rooms(root):
    root = Path(root)
    rooms = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if imgs:
            rooms[d.name] = imgs
    if not rooms:  # a single folder of photos = one room
        imgs = sorted(f for f in root.iterdir() if f.suffix.lower() in IMG_EXT)
        if imgs:
            rooms[root.name] = imgs
    return rooms


def _load(paths):
    imgs = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"cannot read image {p} (HEIC is not supported by OpenCV: export as JPG)")
        imgs.append(im)
    return imgs


def process_room(paths, name, estimator, prior_rel=0.08, log=print):
    from .lidar_tier import process_lidar
    imgs = _load(paths)
    f35 = exif_f35(paths[0]) or F35_DEFAULT
    cap = build_capture(imgs, estimator, f35=f35, root=Path(paths[0]).parent, look_back=len(imgs), log=log)
    n = cap.n_frames
    res, pts = process_lidar(cap, name, stride=1, n_sub=2 if n >= 4 else 1, drift=False,
                             chunk_size=max(n, 1), split_rooms=False, tier="photo", prior_rel=prior_rel)
    res["meta"]["tier_info"] = cap.info
    res["meta"]["caveats"].append(
        f"Photo tier: {cap.info['frames_tracked']}/{cap.info['frames_in']} photos registered. "
        "Coverage beyond the registered photos is not reconstructed; the footprint is a lower bound where walls were not seen.")
    return res, pts


def process_property(root, name, estimator, connections=None, log=print):
    """Whole-property stitched plan from per-room photo folders."""
    from .schema import empty_result
    from .stitch import place_rooms
    rooms = list_rooms(root)
    results = {}
    for rname, paths in rooms.items():
        log(f"room '{rname}': {len(paths)} photos")
        try:
            res, _ = process_room(paths, rname, estimator, log=log)
            results[rname] = res
        except Exception as e:
            log(f"  room '{rname}' failed: {e}")
    if not results:
        raise RuntimeError("no room could be reconstructed from the photos")
    out = empty_result(name, "photo")
    out["rooms"], out["adjacency"], pmeta = place_rooms(results, connections)
    out["meta"] = dict(placement=pmeta, rooms_failed=[r for r in rooms if r not in results],
                       caveats=["Photo tier: room placement uses doorway matching; unmatched rooms are placed "
                                "by assumption and marked."],
                       intervals_status="provisional (8% a-priori floor, uncalibrated)")
    return out
