# Capture protocol (one page, no engineering needed)

Pick by phone. **iPhone Pro (has LiDAR): use A.** Any other iPhone 15 or newer: use B (best) or C.
Whatever you pick: switch on every light, open interior doors, keep people and pets out of frame,
and stay away from mirrors, big windows in direct view, wet-shiny floors and dark rooms.

## A. LiDAR walk (Pro iPhones) - 2 to 4 minutes
1. Install **Stray Scanner** (free, App Store). Open it, tap record.
2. Walk slowly (a slow stroll). Point at walls, floor and **ceiling** in every room; tilt up at the middle of each room.
3. Pause one second in each doorway. End where you started. Tap stop.
4. In the app tap the recording, then **Share** and save the folder to Files. Zip it and AirDrop or cable it to the computer.
5. Unzip into `data/<name>/`. It must contain `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `imu.csv`, `camera_matrix.csv`.
6. Run: `python run.py --tier lidar --input data/<name> --out out/<name>`

## B. Video walk (any iPhone 15+) - 1 to 2 minutes
1. Native **Camera** app, **Video**, 4K at 30 fps, phone upright. No zoom, no cinematic mode.
2. Walk slowly and **turn slowly** (about a quarter turn every 3 seconds). Fast turns blur the picture and lose the track.
3. Keep floor and walls in view, never a blank wall alone. Pause one second in doorways. End where you started.
4. AirDrop the clip to the computer. Put it at `data/<name>/clip.mp4`.
5. Run: `python run.py --tier video --input data/<name> --out out/<name>`

## C. Photos (any iPhone 15+) - 4 to 8 photos per room
1. Settings > Camera > Formats > **Most Compatible** (gives JPG; HEIC is not read).
2. Stand in the middle of the room. Take a photo, turn about 45 degrees, repeat so **each photo overlaps the last by half**.
   Show the floor edge and corners. Add one photo from the doorway. No zoom, no portrait mode, no filters.
3. One folder per room, named for the room: `photos/<house>/kitchen/`, `photos/<house>/hall/` ...
4. Optional `photos/<house>/connections.json`: `[["kitchen","hall"],["hall","bedroom"]]` (rooms that share a door).
   Without it, rooms are assumed to connect in folder-name order and the plan says so.
5. AirDrop the folders. Run: `python run.py --tier photo --input photos/<house> --out out/<house>`

## What you will see
The tool writes `plan.png` and `plan.json`. If it prints **LOW COVERAGE**, too few frames lined up: the numbers are
lower bounds, not dimensions. Redo the capture slower, with more overlap and more texture in view.

*Status: written from the sample data's structure and our measured failure modes. A non-engineer has not yet followed it on a device.*
