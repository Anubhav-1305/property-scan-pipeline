# Device matrix

Accuracy column: **not measured against ground truth for any tier.** No tape or laser measurements exist for the
samples and the author captured none. Status describes what has actually been run.

| Tier | Hardware | Capture tool | Pipeline status | Accuracy honestly delivered |
|---|---|---|---|---|
| LiDAR | Pro-class iPhone (LiDAR scanner), iPhone 12 Pro or newer Pro/Pro Max | Stray Scanner (stock, free) | Runs on the 3 organizer samples (Stray Scanner exports); never run on a capture the author made | Unknown. Internal consistency only: wall sharpness x1.01-1.14 with drift correction on; interval half-width about +-1% to +-3% on area (spread across frame subsets plus a sensor floor). |
| Video | Any iPhone 15 or newer (no LiDAR needed) | Native Camera | Runs end to end with the real depth model on a sample clip (sideways-stored) | Poor. Footprint 73.5% below the LiDAR tier on the one clip tested; result is flagged LOW CONFIDENCE (23% of frames registered). 3% gate not met. |
| Photo | Any iPhone 15 or newer | Native Camera (JPG) | Runs; 5 of 6 close-together stills registered; non-overlapping stills fail loudly | Not measured. Single-room footprint is a lower bound; multi-room placement tested on synthetic rooms only. |

The author did not run any tier on a phone capture. Every result comes from the organizers' LiDAR samples
(video and photo tiers use only the RGB video inside them). Which iPhone models the author has access to is not stated here.
