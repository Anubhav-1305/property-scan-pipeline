# Phone-capture property scanning pipeline: technical report

**Status in one paragraph.** The LiDAR tier runs end to end on the three organizer samples. The video and photo tiers run but are not accurate: on the one clip tested the video footprint was 73.5% below the LiDAR tier's, and any output built from thin coverage is now flagged LOW CONFIDENCE with wide lower-bound intervals. No tape or laser ground truth exists for the samples and none was captured, so **no accuracy against truth is claimed for any tier and no gate is claimed as met**. Damage detection, scope items, the head-to-head app comparison and the organizers' JSON schema are not done (*reports/compliance_matrix.md*).


## 1. Architecture

One geometry core serves all tiers; the tiers differ only in how depth and camera poses are obtained. **Flow:** capture -> depth + poses -> fuse into a z-up point cloud (2 cm voxels, chunked to bound memory) -> drift correction -> floor and ceiling levels -> Manhattan wall angle -> footprint raster -> rectilinear polygon -> wall snapping -> room segmentation and adjacency -> openings -> intervals -> JSON and PNG.

**Key choices.** (a) Buildings are treated as Manhattan: walls are axis-aligned after one estimated rotation, which makes outlines clean and drift correctable but fails on curved or diagonal walls. (b) The floor is the strongest horizontal peak below the camera; a ceiling is accepted only if it is a sharp peak covering a floor-sized area (this rejects a scan that merely stops at some height). (c) Each footprint edge is moved onto the strongest wall-point line within 40 cm, so the outline follows measured walls rather than the occupancy raster. (d) Ceiling height is reported only when a ceiling plane was seen; otherwise it is null. (e) All code runs on a laptop CPU with no calls to our own infrastructure.


## 2. Tier design and device matrix

| Tier | Hardware | Depth | Poses | Status (no ground truth) |
|---|---|---|---|---|
| LiDAR | Pro-class iPhone; Stray Scanner export | sensor depth + confidence map | ARKit odometry | Runs on 3 samples: 41.0, 67.4 and 72.2 m2 footprints; ceiling 3.07 m on the ceiling scan |
| Video | any iPhone 15+, native Camera | Depth Anything V2 Metric Indoor Small (predicted) | RGB-D visual odometry (ORB + PnP) | Runs; footprint 73.5% below LiDAR tier; flagged LOW CONFIDENCE |
| Photo | any iPhone 15+, JPG stills | same model | same odometry between overlapping stills | Runs; 5 of 6 close stills registered; lower-bound footprint; rooms joined by matching doorways (synthetic tests only) |

Details are in *reports/device_matrix.md*. The video tier reads only the mp4; no depth, ARKit or IMU file is touched. The author ran no tier on a self-captured phone recording.


## 3. Drift handling

Handheld odometry drifts, so one wall seen early and late lands in two places. *pipeline/drift.py* anchors the trajectory to the building in three rigid steps per chunk of the scan: (1) rotate the chunk about its centroid until its wall points align with the global Manhattan direction (within 8 degrees); (2) shift it along the two wall axes to agree with the wall-plane positions of all other chunks (clipped to 0.5 m total, accepted only if the cross-correlation improves by 15%); (3) shift its floor height to the global floor. The whole correction is kept only if wall sharpness improves overall. Every applied value is logged in *plan.json*. This is not a pose graph or loop closure.

| Capture | Wall sharpness, on vs off | Outline perimeter | Footprint | Largest yaw fix |
|---|---|---|---|---|
| single_room | x1.011 | +1.0% | +0.6% | 1.0 deg |
| single_scan_floor_only | x1.139 | -1.9% | +0.4% | 4.75 deg |
| single_scan_with_ceiling | x1.055 | -4.4% | +4.5% | 2.5 deg |

Wall sharpness is points per 2 cm bin along the wall axes (higher means thinner wall lines). These are internal-consistency metrics, not error against truth. The shift correction reached its 0.5 m clip on all three scans, so larger drift is not recovered. Side-by-side plans: *results/ablation/*/side_by_side.png*.


## 4. Error budget

| Source | Size | Basis | Affects |
|---|---|---|---|
| LiDAR depth noise | millimetres to ~1 cm per point; averaged over many points | assumed from sensor class; not measured here | all lengths, ceiling |
| Pose drift | up to the 0.5 m clip; corrected only partly | measured: shifts hit the clip on all samples | footprint, adjacency |
| Voxel and wall-snap resolution | 2 cm | design constant | lengths, opening widths |
| Passage width raster | 5 cm | design constant | room-to-room openings |
| Manhattan assumption | unbounded for non-rectilinear rooms | known limit | footprint shape |
| Focal length when EXIF missing (26 mm equiv.) | about +-6% lateral scale | assumed from sample's true value 28.8 mm | video/photo lengths |
| Monocular depth scale error | unknown; lengths scale linearly, areas quadratically | not measured | video/photo everything |
| Coverage (frames lost) | video: 77% of keyframes lost on the clip tested | measured | footprint (lower bound) |


## 5. Calibration analysis

Intervals come from three parts combined in quadrature: (a) the spread of the geometry across three interleaved frame subsets of the same capture; (b) a fixed sensor floor (1 cm or 0.5% on lengths, 1 cm on heights, 2 cm plus 1% on openings); (c) an a-priori tier term of 3% for video and 8% for photo, taken from the gate values, not fitted. Widening for thin point evidence is added per wall. **None of this is calibrated**: calibration needs measured truth, so interval coverage probability is unknown at every tier. The split-subset spread is not the same-room-twice repeatability the brief asks for.

**Confident-garbage guard.** When fewer than 80% of frames register, *lower_bound_ci* makes the interval asymmetric: the upper end of area is value / coverage (lengths: value / sqrt(coverage)), the room is marked confidence = low, the plan image says LOW CONFIDENCE, and the CLI warns. On the video clip (10.86 m2 at 23% coverage) the interval's upper end is about 47 m2, which would have covered the 40.97 m2 LiDAR-tier result. This is a geometric heuristic checked on one capture; it is not a statistical calibration.


## 6. Fix loop

**Failing gate:** video-tier footprint on single_room, 73-79% below the LiDAR tier (a pseudo-reference, not a tape measurement). **Hypotheses (declared before fixing):** the tracker copies a pose when it loses a frame and later frames chain through it (H1); gravity is taken from a plane fit with no check that it lies below the camera (H2). **Shipped:** segmented tracking that never guesses, a physical-plausibility check on every step, relocalisation against earlier frames, floor-below-camera gravity, and sharpest-frame-per-window keyframes (the old blur filter had deleted frames 204 to 268 of the clip, a 64-frame gap with no overlap).

| Depth | Mode | Frames kept | Path error (m) | Gravity err median (deg) | Footprint (m2) | vs LiDAR |
|---|---|---|---|---|---|---|
| real model | before | 344/350 | 2.394 | 79.4 | 8.65 | -78.9% |
| real model | after | 99/428 | 0.348 | 24.8 | 10.86 | -73.5% |
| simulated | before | 344/350 | 0.858 | 52.3 | 4.54 | -88.9% |
| simulated | after | 99/428 | 0.037 | 8.7 | 2.23 | -94.5% |

**Result:** H1 confirmed (path error 7x to 23x better) and H2 partly (gravity 3x to 6x better, the 5 degree target missed). The footprint gate did not move toward pass. **Both predictions were badly wrong** (simulated: predicted within 15%, got -94.5%; real model: predicted -40% to +15%, got -73.5%). The correct prediction was that the 3% gate would not pass. **Post-mortem:** I assumed tracking was the only bottleneck. Fixing it exposed coverage: during a fast turn about 12 consecutive keyframes share no features with their neighbours, a video file carries no gyroscope track to bridge them, so only the largest tracked segment (23% of frames) is kept and the plan covers a fraction of the room. Accurate poses over a quarter of the walk beat inaccurate poses over all of it on path error, not on footprint. Not built: a depth-based (ICP) bridge across blind stretches. Evidence is limited to one capture, simulated or predicted depth, and a LiDAR-tier pseudo-reference. Regenerate: *fixloop/before_after.md*.


## 7. Reproduction

*bash scripts/regenerate_all.sh* reruns tests, the three LiDAR runs, the drift ablation, the simulated before/after fix loop and the benchmark report from the sample data; *--with-model* adds the real-model runs (depth cached so before and after replay identical input). Weights are fetched by *scripts/fetch_weights.py*, never committed.


## 8. Known failure modes

**Mirrors and glass.** A mirror returns depth to the reflected scene, so a phantom room can appear behind the wall; windows and glass doors return depth through or off the glass. LiDAR confidence filtering (confidence 2 only) removes some but not all of this; monocular depth is worse on glass. Not tested on data. **Wet-look and glossy floors** cause specular holes in depth and weaken floor-plane fits. **Low light** leaves LiDAR depth usable but degrades feature tracking, so poses and the video/photo tiers suffer first. **Fast turns and blank walls** are the dominant video failure (measured above). **Open-plan spaces and wide openings** are not split: room segmentation separates rooms only at doorways narrower than about 1 m, and found one region on every real sample. **Furniture** produces wall-height points that corrupt interior-wall detection (an interior-wall variant of room splitting was tried and dropped). **High ceilings:** fusion keeps depth up to 4.5 m, so ceilings farther than that from the camera are missed. **Non-rectilinear rooms** are forced onto axes. **Overfitting risk:** thresholds (ceiling coverage 0.35, cliff ratio 8, erosion 0.5 m) were set on three samples.


## 9. Not done

Damage regions, concealed-damage rules and scope items; the benchmark set (multi-room capture, staged damage, all three tiers on the same rooms, repeat capture, laser ground truth); the head-to-head against a consumer app; the published JSON schema; an own iOS capture app; a timed clean-machine install; a non-engineer trial of the capture protocol.
