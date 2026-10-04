#!/usr/bin/env bash
# Regenerate every reported number from raw inputs. Needs data/<capture>/ (see README).
#   bash scripts/regenerate_all.sh               # no model download needed
#   bash scripts/regenerate_all.sh --with-model  # also real-depth-model fix-loop runs
set -euo pipefail
python -m pytest -q tests
for z in single_room single_scan_floor_only single_scan_with_ceiling; do
  python run.py --tier lidar --input "data/$z" --out "out/$z"
  python benchmark/ablation_drift.py "data/$z" results/ablation
  mkdir -p "results/samples/$z"
  cp "results/ablation/$z/drift_on.json" "results/samples/$z/plan.json"
  cp "results/ablation/$z/drift_on.png" "results/samples/$z/plan.png"
done
python benchmark/fixloop_run.py data/single_room --mode before --depth sim
python benchmark/fixloop_run.py data/single_room --mode after  --depth sim
if [ "${1:-}" = "--with-model" ]; then
  python benchmark/fixloop_run.py data/single_room --mode before --depth model
  python benchmark/fixloop_run.py data/single_room --mode after  --depth model
fi
python benchmark/fixloop_run.py --table > fixloop/before_after_table.md
python benchmark/run_benchmark.py
echo "done"
