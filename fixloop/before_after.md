# Fix loop: before vs after (video tier, `single_room`)

Regenerate (sandbox / no model needed, simulated depth):

    python benchmark/fixloop_run.py data/single_room --mode before --depth sim
    python benchmark/fixloop_run.py data/single_room --mode after  --depth sim
    python benchmark/fixloop_run.py --table

Regenerate with the real depth model (depth cached so both runs see identical depth):

    python benchmark/fixloop_run.py data/single_room --mode before --depth model
    python benchmark/fixloop_run.py data/single_room --mode after  --depth model

Diff of the shipped fix: `fixloop/fix_vo.diff`, `fixloop/fix_video_tier.diff`
(against `fixloop/before_snapshot/`). `mode="before"` still runs the original code.

## Results

| depth | mode | tracked | path error (m) | gravity err median/max (deg) | footprint (m2) | vs LiDAR ref |
|---|---|---|---|---|---|---|
| model (real) | before | 344/350 | 2.394 | 79.4 / 84.8 | 8.65 | -78.9% |
| model (real) | after | 99/428 | 0.348 | 24.8 / 44.9 | 10.86 | -73.5% |
| sim | before | 344/350 | 0.858 | 52.3 / 126.6 | 4.54 | -88.9% |
| sim | after | 99/428 | 0.037 | 8.7 / 10.3 | 2.23 | -94.5% |

Real-model rows were run on the author's machine (CPU, Depth Anything V2 metric indoor
small); sim rows in the sandbox. Within each depth source, before and after see identical
depth. The earlier 10.5 m2 figure in `FIX_DECLARATION.md` came from `run.py` with a
different focal-length assumption (26 mm vs 28.8 mm here); the table above is the
controlled comparison.

## Outcome against the declaration

| Prediction (made before the fix) | Result |
|---|---|
| Footprint within +-15% of LiDAR tier (simulated depth) | **Wrong: -94.5%** |
| Real model: footprint between -40% and +15% | **Wrong: -73.5%** |
| Path error below 0.4 m (simulated depth) | Met: 0.037 m (real model: 0.348 m, also under 0.4) |
| Gravity error below 5 deg (simulated depth) | Missed: 8.7 deg (real model: 24.8 deg) |
| "The 3% gate will not pass" | Correct |

Real-model movement: path error 2.394 to 0.348 m (6.9x better), gravity 79.4 to 24.8 deg
median (3.2x better), footprint error only -78.9% to -73.5% (8.65 to 10.86 m2). The gate
(video footprint) is not met and barely moved. The prediction was badly wrong in both
depth settings.

## Post-mortem

What the fix got right. The old tracker copied a pose when it lost a frame and later
frames chained through it, which put wrong poses into the cloud. That is confirmed by the
path error falling 6.9x with the real model and 23x with simulated depth, and by the
gravity error falling 3.2x to 6x. The new tracker never uses a guess.

Why the gate did not move. The footprint is limited by coverage, not by pose accuracy:

1. Only 99 of 428 keyframes (23%) survive. About 12 consecutive keyframes during a fast
   turn share no features with their neighbours, and a video file has no gyroscope to
   carry the pose across. The tracker starts a new segment there and keeps only the
   largest one, so most of the room is never reconstructed. A footprint built from 23%
   of the walk is small however accurate those poses are.
2. The original blur filter also deleted a run of frames and left a 64-frame time gap
   (video frames 204 to 268). Fixed: the sharpest frame in each time window is now
   used (`tests/test_keyframes.py`). This removed one cause of lost tracking but not the
   fast-turn blind stretch.
3. Gravity is still 24.8 deg off with the real model (8.7 deg with simulated depth). The
   real model's floor is less planar than LiDAR depth, so the floor-plane fit tilts. A
   tilted floor skews wall lengths. I could not separate this from depth-scale error.

What I got wrong. I wrote the prediction assuming tracking was the only bottleneck.
Fixing tracking exposed the coverage limit. I also underestimated how much real-model
depth error would hurt the gravity fit.

Not done, and what would help: a depth-based fallback (ICP between depth clouds) to bridge
blind stretches; the capture protocol's slow-walk rule, which the sample clip does not
follow; gravity from a second cue (wall verticality) to cap the tilt. A real iPhone video
file carries no usable gyroscope track.

Limits of this evidence: one capture; footprint compared to the LiDAR tier, not to a tape
measurement; ATE and gravity use the phone's own ARKit poses as a proxy reference.
