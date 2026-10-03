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
        res, pts = process_lidar(cap, name, stride=args.stride)
        render_plan(res, out / "plan.png", points=pts[:, :2])
    else:
        raise SystemExit(f"tier '{args.tier}' is not implemented yet (planned: Phase 3)")

    (out / "plan.json").write_text(json.dumps(res, indent=2))
    r = res["rooms"][0]
    h = r["ceiling_height_m"]["value"]
    print(f"[{name}] area {r['floor_area_m2']['value']:.1f} m2 | "
          f"ceiling {'n/a' if h is None else f'{h:.2f} m'} | "
          f"{len(r['openings'])} openings | {res['meta']['runtime_s']} s")
    print(f"wrote {out/'plan.json'} and {out/'plan.png'}")


if __name__ == "__main__":
    main()
