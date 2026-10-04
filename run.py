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
    else:
        raise SystemExit(f"tier '{args.tier}' is not implemented yet (planned: Phase 3)")

    (out / "plan.json").write_text(json.dumps(res, indent=2))
    m = res["meta"]
    h = res["rooms"][0]["ceiling_height_m"]["value"]
    n_open = sum(len(r["openings"]) for r in res["rooms"])
    print(f"[{name}] {len(res['rooms'])} rooms | total {m['total_footprint_m2']:.1f} m2 | "
          f"ceiling {'n/a' if h is None else f'{h:.2f} m'} | {n_open} openings | "
          f"drift {'on' if m['drift']['enabled'] else 'off'} | {m['runtime_s']} s")
    print(f"wrote {out/'plan.json'} and {out/'plan.png'}")


if __name__ == "__main__":
    main()
