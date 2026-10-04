"""One command per capture.

    python run.py --tier lidar --input data/single_room --out out/single_room
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Phone capture -> dimensioned floor plan")
    p.add_argument("--tier", choices=["lidar", "video", "photo"], required=True)
    p.add_argument("--input", required=True, help="capture folder")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--rotate", choices=["cw", "ccw", "180"], help="rotate video frames upright (only for raw sensor-orientation clips such as the samples)")
    p.add_argument("--no-drift", action="store_true", help="disable drift correction (ablation)")
    p.add_argument("--stride", type=int, default=8, help="use every Nth frame (lidar)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).name

    if args.tier == "lidar":
        from pipeline.io_stray import load_capture
        from pipeline.lidar_tier import process_lidar
        from pipeline.render import render_plan
        cap = load_capture(args.input)
        res, pts = process_lidar(cap, name, stride=args.stride, drift=not args.no_drift)
        render_plan(res, out / "plan.png", points=pts[:, :2])
    elif args.tier == "video":
        from pipeline.depth_models import DepthAnythingV2Metric
        from pipeline.render import render_plan
        from pipeline.video_tier import process_video
        src = Path(args.input)
        video = src if src.is_file() else next(src.rglob("rgb.mp4"), None) or next(iter(sorted(src.rglob("*.mp4"))), None)
        if video is None:
            raise SystemExit(f"no video found in {src}")
        print(f"video tier: reading only {video} (no depth/pose files are used)")
        res, pts = process_video(video, name, DepthAnythingV2Metric(), rotate=args.rotate)
        render_plan(res, out / "plan.png", points=pts[:, :2])
    else:  # photo
        import json as _json
        from pipeline.depth_models import DepthAnythingV2Metric
        from pipeline.photo_tier import process_property
        from pipeline.render import render_plan
        conn = None
        cfile = Path(args.input) / "connections.json"
        if cfile.exists():
            conn = _json.loads(cfile.read_text())
        res = process_property(args.input, name, DepthAnythingV2Metric(), connections=conn)
        render_plan(res, out / "plan.png")

    (out / "plan.json").write_text(json.dumps(res, indent=2))
    m = res["meta"]
    h = res["rooms"][0]["ceiling_height_m"]["value"]
    n_open = sum(len(r["openings"]) for r in res["rooms"])
    total = m.get("total_footprint_m2") or sum(r["floor_area_m2"]["value"] for r in res["rooms"])
    print(f"[{name}] tier={args.tier} | {len(res['rooms'])} rooms | total {total:.1f} m2 | "
          f"ceiling {'n/a' if h is None else f'{h:.2f} m'} | {n_open} openings")
    print(f"wrote {out/'plan.json'} and {out/'plan.png'}")


if __name__ == "__main__":
    main()
