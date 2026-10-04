"""Build reports/benchmark.md from the result files in results/.

    python benchmark/run_benchmark.py

Reads (does not recompute): results/samples/*/plan.json (LiDAR tier),
results/ablation/*/ablation.json (drift), results/fixloop/*.json (video fix loop).
Regenerate those with run.py, benchmark/ablation_drift.py and benchmark/fixloop_run.py.

HONESTY NOTE: no tape/laser ground truth exists for the organizers' sample scans,
and the author captured none. So this report contains NO accuracy-vs-truth numbers.
It reports internal consistency, interval widths, and tier-to-tier agreement.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def fmt(ci):
    if ci["value"] is None:
        return "not captured"
    return f"{ci['value']:.2f} (+-{ci['half_width']:.2f})"


def main():
    L = ["# Benchmark report", "",
         "**No ground truth.** The sample scans came without tape or laser measurements and no",
         "own captures were made, so nothing below is accuracy against truth. Gates are NOT claimed.",
         "", "## LiDAR tier on the organizers' samples (drift correction on)", "",
         "| Capture | Rooms found | Footprint m2 (95% interval) | Ceiling height m | Openings | Runtime s |",
         "|---|---|---|---|---|---|"]
    for p in sorted((R / "samples").glob("*/plan.json")):
        d = json.loads(p.read_text())
        m = d["meta"]
        a = sum(r["floor_area_m2"]["value"] for r in d["rooms"])
        hw = sum(r["floor_area_m2"]["half_width"] for r in d["rooms"])
        h = d["rooms"][0]["ceiling_height_m"]
        n_open = sum(len(r["openings"]) for r in d["rooms"])
        L.append(f"| {p.parent.name} | {len(d['rooms'])} | {a:.1f} (+-{hw:.1f}) | {fmt(h)} | {n_open} | {m['runtime_s']} |")
    L += ["", "Intervals are provisional (never calibrated against measured truth).", "",
          "## Drift ablation (internal consistency only)", "",
          "| Capture | Wall sharpness on/off | Perimeter change | Footprint change |", "|---|---|---|---|"]
    for p in sorted((R / "ablation").glob("*/ablation.json")):
        d = json.loads(p.read_text())["delta"]
        L.append(f"| {p.parent.name} | x{d['wall_sharpness_ratio_on_over_off']:.3f} | "
                 f"{d['perimeter_change_pct']:+.1f}% | {d['footprint_area_change_pct']:+.1f}% |")
    L += ["", "## Repeatability", "",
          "Not measured as specified (no room was captured twice). The intervals above come from",
          "re-running the geometry on interleaved frame subsets of one capture, which is a",
          "split-half consistency check, not the same-room-twice gate.", "",
          "## Video and photo tiers", "",
          "See `fixloop/before_after.md`. Photo tier: ran end to end on 6 close-together stills",
          "cut from a sample clip (5/6 registered) and failed loudly on non-overlapping stills.",
          "No footprint accuracy is claimed for either tier.", "",
          "## Head-to-head vs a consumer scanning app", "",
          "Not done: needs a Pro-class iPhone and an installed app on the same rooms.", ""]
    (ROOT / "reports" / "benchmark.md").write_text("\n".join(L))
    print("wrote reports/benchmark.md")


if __name__ == "__main__":
    main()
