# Fix declaration (written BEFORE the fix was implemented)

## 1. Worst-performing gate in my own benchmark, with the failing number

**Video-tier footprint on `single_room`** (gate: wall lengths within +-3% at the video tier).

| Run | Footprint | LiDAR-tier result on the same scan | Error |
|---|---|---|---|
| Real depth model (Depth Anything V2 metric indoor small), author's machine | 10.5 m2 | 40.7 m2 | **-74%** |
| Simulated depth (LiDAR + 0% scale error, 0% noise), sandbox | 6.9 m2 | 40.7 m2 | -83% |
| Simulated depth, earlier tracker version | 24.5-26.4 m2 | 40.7 m2 | -36 to -40% |

Caveat on the reference: the 40.7 m2 is the LiDAR tier's answer, not a tape or laser
measurement. No ground-truth measurements exist for the sample scans. The gate is
therefore scored against a pseudo-reference, and the numbers show a gross failure,
not a precise error.

Second failing number from the same run: path error (ATE RMSE against the phone's
own ARKit poses) of 1.1-1.3 m over a 13.8 m walk.

## 2. Root-cause hypothesis and evidence

Hypothesis: the **visual odometry is the root cause**, in two linked ways.

H1. When a frame finds no match, the tracker "holds" (copies the previous pose) and
later frames are then allowed to chain through that wrong pose.
- Evidence: per-step relative motion against ARKit is accurate when tracking succeeds
  (median 0.5 deg rotation, 0.9 cm translation per step), but 6-9 frames per clip find
  zero matches, and one bridged step was off by ~130 deg.
- Evidence: path error stays at ~1.1-1.3 m while tracked-frame count rises from 24/156
  to 342/349, so more tracked frames alone do not fix it; the bridged jumps do the damage.

H2. Gravity (the "up" direction) is estimated from the dominant horizontal plane of
the drifted point cloud, with an image-up prior and no check that the plane is below the
camera.
- Evidence: gravity error against ARKit gravity was 63 deg on the sideways clip and
  14 deg after rotating frames upright, with only 3-8% of points as plane inliers.
  A tilted floor shrinks and skews every wall length and the footprint.

What I could not separate: how much of the real-model result comes from depth-model
error versus these two causes. The simulated-depth runs isolate the tracker/gravity
causes; the real-model run mixes in depth error.

## 3. The fix I intend to ship, and the numbers I predict after it

Fix (in `pipeline/vo.py`, switchable with `mode="before"|"after"`):
1. Lost frames are dropped, never held or referenced; later frames match against the
   last valid frames only.
2. A pose is accepted only if it is physically plausible (step < 1.0 m and < 35 deg) and
   it is the best of several candidate references (most inliers).
3. Gravity is taken from the largest near-horizontal plane that sits 0.8-2.2 m *below*
   the camera path, then refit on its inliers.

Predictions (made before running the fix):
- Simulated depth, 0% scale error: footprint within +-15% of the LiDAR-tier result,
  path error below 0.4 m, gravity error below 5 deg.
- Real depth model: improves but does not reach the gate. Footprint error between
  -40% and +15%, because depth-model error remains. I do **not** predict the 3% gate
  will pass.
- If the simulated-depth gravity error is still above 5 deg after the fix, hypothesis
  H2 was wrong or incomplete, and the post-mortem will say so.

---

## Outcome (appended after the fix was run; the text above is unchanged)

Both footprint predictions were badly wrong: simulated depth -94.5% (predicted +-15%) and
real model -73.5% (predicted -40% to +15%). Pose results improved a lot: path error
0.858 to 0.037 m (sim) and 2.394 to 0.348 m (real model); gravity error 52.3 to 8.7 deg
(sim) and 79.4 to 24.8 deg (real model), but the 5 deg target was missed. H1 (held frames
poison later poses) was confirmed. H2 (gravity) was partly confirmed. The hypothesis was
incomplete: coverage lost to blind stretches limits the footprint more than pose error
does. The "3% gate will not pass" prediction was correct. Full post-mortem:
`fixloop/before_after.md`.
