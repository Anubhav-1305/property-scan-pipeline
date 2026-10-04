"""Pull N evenly spaced stills from a video into a per-room photo folder.

    python scripts/make_photo_set.py clip.mp4 photos/room1 --n 6

Convenience for rehearsing the photo tier when you only have a video. Real photo
captures (taken by hand) are what the photo tier is meant for.
"""
import argparse
from pathlib import Path

import cv2

p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("out"); p.add_argument("--n", type=int, default=6)
a = p.parse_args()
cap = cv2.VideoCapture(a.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
Path(a.out).mkdir(parents=True, exist_ok=True)
for k, i in enumerate(range(0, n, max(1, n // a.n))[: a.n]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if ok:
        cv2.imwrite(str(Path(a.out) / f"{k:02d}.jpg"), f)
print("wrote", a.n, "stills to", a.out)
