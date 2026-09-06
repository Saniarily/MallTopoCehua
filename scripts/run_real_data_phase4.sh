#!/usr/bin/env bash
# Round 3 (Phase 4) experiments on the REAL data (Mac M4 Pro, conda env `mallranker`).
#
#   bash scripts/run_real_data_phase4.sh                 # corpus path read from configs/data/legacy(.local).yaml sharegpt_json
#   SKIP_S1=1 bash scripts/run_real_data_phase4.sh       # stage-2 only (AR-GNN training + 4-way comparison)
#   CORPUS=/path/to/sharegpt_data.json bash scripts/run_real_data_phase4.sh   # explicit corpus
#
# Optional env: S2_LIMIT=600 (stage-2 holdout size to evaluate), DEVICE=auto|cpu|mps for the AR-GNN
#               (auto = CUDA else CPU; MPS is opt-in and enables the CPU fallback for unsupported ops),
#               SKIP_S1=1 / SKIP_S2=1. run_ablation resumes: already-finished (variant, seed) cells are reused.
# Everything is written under outputs/experiments/*_r3 so round-2 results stay untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
CORPUS="${CORPUS:-}"; S2_LIMIT="${S2_LIMIT:-600}"; DEVICE="${DEVICE:-auto}"
S1_OUT=outputs/experiments/real_r3; mkdir -p "$S1_OUT"

python scripts/prepare_data.py --config configs/data/legacy.yaml

if [[ "${SKIP_S1:-0}" != "1" ]]; then
  echo "== [S1-A] type-conditional quality model E[score | conditions, type] (5 seeds) =="
  python scripts/evaluate_type_recommender.py --config configs/stage1/base.yaml --seeds 42 43 44 45 46 \
    --out "$S1_OUT/type_recommender"

  echo "== [S1-B] model comparison incl. deep_residual (3 seeds; same grouped split as round 2) =="
  python scripts/run_ablation.py --config configs/ablations/stage1_model_comparison.yaml --out-dir "$S1_OUT/model_comparison"

  echo "== [S1-C] deep ranker ingredient ablation (3 seeds) =="
  python scripts/run_ablation.py --config configs/ablations/stage1_deep_ablation.yaml --out-dir "$S1_OUT/deep_ablation"

  echo "== [S1-D] prototype fidelity for deep_residual vs extra_trees (label-free protocol) =="
  python scripts/evaluate_fidelity.py --config configs/stage1/extra_trees.yaml \
    --configs configs/stage1/deep_residual.yaml configs/stage1/quality_oracle.yaml --out "$S1_OUT/fidelity"
fi

if [[ "${SKIP_S2:-0}" != "1" ]]; then
  # corpus: $CORPUS env > configs/data/legacy.local.yaml > configs/data/legacy.yaml (sharegpt_json)
  if [[ -z "$CORPUS" ]]; then
    CORPUS=$(python - <<'PY'
import sys; sys.path.insert(0, "src")
from mall_space_planner.utils import resolve_config
print(resolve_config("configs/data/legacy.yaml").get("sharegpt_json") or "")
PY
)
  fi
  if [[ -z "$CORPUS" || ! -f "$CORPUS" ]]; then
    echo "!! stage-2 corpus not found (CORPUS='$CORPUS'). Set CORPUS=/path/to/sharegpt_data.json or sharegpt_json in configs/data/legacy.local.yaml"; exit 2
  fi
  echo "== stage-2 corpus: $CORPUS =="
  S2_OUT=outputs/experiments/stage2_eval_r3
  echo "== [S2-A] train AR-GNN on corpus minus last 600 (holdout) =="
  python scripts/train_stage2.py --config configs/stage2/ar_gnn.yaml --corpus "$CORPUS" \
    --override "stage2.generator.params.checkpoint=null" "stage2.generator.params.device=$DEVICE"

  echo "== [S2-B] generator comparison on the SAME 600 held-out skeletons (+ ground-truth reference row) =="
  python scripts/evaluate_stage2.py --config configs/stage2/rule_baseline.yaml --corpus "$CORPUS" --limit "$S2_LIMIT" --ground-truth \
    --override "eval_output_dir=$S2_OUT" || echo "!! ground truth failed"
  for CFG in rule_baseline search_baseline; do
    python scripts/evaluate_stage2.py --config "configs/stage2/$CFG.yaml" --corpus "$CORPUS" --limit "$S2_LIMIT" \
      --override "eval_output_dir=$S2_OUT" || echo "!! $CFG failed"
  done
  for CFG in ar_gnn ar_gnn_greedy ar_gnn_bestof16; do   # only learned generators accept `device`
    python scripts/evaluate_stage2.py --config "configs/stage2/$CFG.yaml" --corpus "$CORPUS" --limit "$S2_LIMIT" \
      --override "eval_output_dir=$S2_OUT" "stage2.generator.params.device=$DEVICE" || echo "!! $CFG failed"
  done
  python - <<'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/experiments/stage2_eval_r3/*/aggregate.json")):
    a = json.load(open(f)); rows.append((os.path.basename(os.path.dirname(f)), a))
keys = ["overall_pass", "node_deviation_pct", "edge_accuracy_pct", "density_deviation_pct", "aspl_deviation_pct", "n_components", "target_edge_recall_pct", "target_edge_precision_pct", "inference_time_s"]
print("| generator | " + " | ".join(keys) + " |"); print("|" + "---|" * (len(keys) + 1))
for n, a in rows: print(f"| {n} | " + " | ".join(f"{a.get(k, float('nan')):.3f}" for k in keys) + " |")
open("outputs/experiments/stage2_eval_r3/table.md", "w").write("| generator | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n" + "\n".join(f"| {n} | " + " | ".join(f"{a.get(k, float('nan')):.3f}" for k in keys) + " |" for n, a in rows) + "\n")
EOF
fi

echo "== report back =="
echo "  $S1_OUT/type_recommender/results.json"
echo "  $S1_OUT/model_comparison/table_test.md   $S1_OUT/deep_ablation/table_test.md"
echo "  $S1_OUT/fidelity/fidelity_summary.csv"
echo "  outputs/experiments/stage2_eval_r3/table.md  (+ per-config aggregate.json)"
