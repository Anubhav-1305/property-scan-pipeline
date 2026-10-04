# Benchmark report

**No ground truth.** The sample scans came without tape or laser measurements and no
own captures were made, so nothing below is accuracy against truth. Gates are NOT claimed.

## LiDAR tier on the organizers' samples (drift correction on)

| Capture | Rooms found | Footprint m2 (95% interval) | Ceiling height m | Openings | Runtime s |
|---|---|---|---|---|---|
| single_room | 1 | 41.0 (+-4.2) | not captured | 0 | 18.2 |
| single_scan_floor_only | 1 | 67.4 (+-0.8) | not captured | 1 | 51.0 |
| single_scan_with_ceiling | 1 | 72.2 (+-14.3) | 3.07 (+-0.01) | 1 | 104.8 |

Intervals are provisional (never calibrated against measured truth).

## Drift ablation (internal consistency only)

| Capture | Wall sharpness on/off | Perimeter change | Footprint change |
|---|---|---|---|
| single_room | x1.011 | +1.0% | +0.6% |
| single_scan_floor_only | x1.139 | -1.9% | +0.4% |
| single_scan_with_ceiling | x1.055 | -4.4% | +4.5% |

## Repeatability

Not measured as specified (no room was captured twice). The intervals above come from
re-running the geometry on interleaved frame subsets of one capture, which is a
split-half consistency check, not the same-room-twice gate.

## Video and photo tiers

See `fixloop/before_after.md`. Photo tier: ran end to end on 6 close-together stills
cut from a sample clip (5/6 registered) and failed loudly on non-overlapping stills.
No footprint accuracy is claimed for either tier.

## Head-to-head vs a consumer scanning app

Not done: needs a Pro-class iPhone and an installed app on the same rooms.
