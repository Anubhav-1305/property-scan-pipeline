"""Drift ablation: run the same capture with drift correction OFF and ON.

    python benchmark/ablation_drift.py data/single_scan_with_ceiling results/ablation

Writes drift_off.png, drift_on.png, side_by_side.png and ablation.json.
No ground truth is used: metrics are internal consistency measures
(wall sharpness, footprint area, outline complexity). Lower perimeter/vertex
count and higher sharpness mean walls from different times in the scan agree.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from pipeline.io_stray import load_capture
from pipeline.lidar_tier import process_lidar
from pipeline.render import render_plan


def metrics(res):
    m = res["meta"]
    outline_polys = [np.array(r["polygon_m"]) for r in res["rooms"]]
    perim = sum(Polygon(p).length for p in outline_polys)
    return dict(
        wall_sharpness=m["drift"]["wall_sharpness"],
        total_footprint_m2=m["total_footprint_m2"],
        outline_perimeter_m=perim,
        outline_vertices=int(sum(len(p) for p in outline_polys)),
        n_rooms=len(res["rooms"]),
        max_abs_yaw_correction_deg=m["drift"]["max_abs_yaw_deg"],
        max_abs_shift_correction_m=m["drift"]["max_abs_shift_m"],
        corrections_rejected=any(c.get("rejected") for c in m["drift"]["chunks"]),
    )


def main(capture, outroot):
    capture = Path(capture)
    name = capture.name
    out = Path(outroot) / name
    out.mkdir(parents=True, exist_ok=True)
    cap = load_capture(capture)
    report, imgs = {}, {}
    for tag, flag in (("drift_off", False), ("drift_on", True)):
        res, pts = process_lidar(cap, name, drift=flag)
        res["title"] = f"{name} - {tag.replace('_', ' ')}"
        render_plan(res, out / f"{tag}.png", points=pts[:, :2])
        report[tag] = metrics(res)
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
        imgs[tag] = plt.imread(out / f"{tag}.png")
    off, on = report["drift_off"], report["drift_on"]
    report["delta"] = dict(
        wall_sharpness_ratio_on_over_off=on["wall_sharpness"] / off["wall_sharpness"],
        footprint_area_change_pct=100 * (on["total_footprint_m2"] - off["total_footprint_m2"]) / off["total_footprint_m2"],
        perimeter_change_pct=100 * (on["outline_perimeter_m"] - off["outline_perimeter_m"]) / off["outline_perimeter_m"],
    )
    (out / "ablation.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a, t in zip(ax, ("drift_off", "drift_on")):
        a.imshow(imgs[t]); a.axis("off"); a.set_title(t.replace("_", " "))
    fig.savefig(out / "side_by_side.png", dpi=110, bbox_inches="tight")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/ablation")
