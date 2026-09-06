#!/usr/bin/env bash
# Round 4 (Mac): complete the stage-2 experiment set for the thesis.
#   bash scripts/run_real_data_round4.sh            # corpus from configs/data/legacy(.local).yaml; resumable (skips finished cells)
# Steps: (A) multi-seed eval of the existing v2 checkpoint + baselines (3 seeds)  (B) AR-GNN ablations (3 trainings)
#        (C) longer/larger model (20 epochs, d=96) + best-of-16  (D) aggregate table with mean ± std.
# Estimated: A ~1.5 h (bestof16 dominates), B ~40 min train + 30 min eval, C ~35 min train + 40 min eval.
set -euo pipefail
cd "$(dirname "$0")/.."
CORPUS="${CORPUS:-$(python - <<'PY'
import sys; sys.path.insert(0, "src")
from mall_space_planner.utils import resolve_config
print(resolve_config("configs/data/legacy.yaml").get("sharegpt_json") or "")
PY
)}"
[[ -f "$CORPUS" ]] || { echo "!! corpus not found: '$CORPUS'"; exit 2; }
S2_OUT=outputs/experiments/stage2_eval_r4; LIMIT="${S2_LIMIT:-600}"; SEEDS="${SEEDS:-0 1 2}"; DEVICE="${DEVICE:-auto}"
echo "== corpus: $CORPUS  seeds: $SEEDS =="

ev() { # cfg, seed, extra...
  local CFG=$1 SEED=$2; shift 2
  python scripts/evaluate_stage2.py --config "configs/stage2/$CFG.yaml" --corpus "$CORPUS" --limit "$LIMIT" --seed "$SEED" "$@" --override "eval_output_dir=$S2_OUT" || echo "!! $CFG seed $SEED failed"
}
tr() { python scripts/train_stage2.py --config "configs/stage2/$1.yaml" --corpus "$CORPUS" --override "stage2.generator.params.checkpoint=null" "stage2.generator.params.device=$DEVICE" || echo "!! train $1 failed"; }

echo "== [A] ground truth + baselines + v2 (3 seeds) =="
ev rule_baseline 0 --ground-truth
[[ -f outputs/checkpoints/stage2/stage2_ar_gnn/ar_gnn.pt ]] || tr ar_gnn
for S in $SEEDS; do
  ev rule_baseline "$S"; ev search_baseline "$S"
  for CFG in ar_gnn ar_gnn_greedy ar_gnn_bestof16; do ev "$CFG" "$S" --override "stage2.generator.params.device=$DEVICE"; done
done

echo "== [B] AR-GNN ablations: order / loss / features (train once, eval 3 seeds) =="
for CFG in ar_gnn_bfs_order ar_gnn_single_label ar_gnn_basic_feats; do
  tr "$CFG"
  for S in $SEEDS; do ev "$CFG" "$S" --override "stage2.generator.params.device=$DEVICE"; done
done

echo "== [C] longer / larger model =="
tr ar_gnn_long
for S in $SEEDS; do ev ar_gnn_long "$S" --override "stage2.generator.params.device=$DEVICE"; ev ar_gnn_long_bestof16 "$S" --override "stage2.generator.params.device=$DEVICE"; done

echo "== [D] aggregate =="
python scripts/aggregate_stage2.py --root "$S2_OUT"
echo "== report back: $S2_OUT/table.md, $S2_OUT/summary.csv, outputs/checkpoints/stage2/*/meta.json =="
