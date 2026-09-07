#!/usr/bin/env bash
# Round 5 (Mac): Stage 2 on the REAL corpus v2 (skeleton *_M_simplified.csv -> complete *_M.csv), replacing the
# LLM-generated ShareGPT corpus used in rounds 1-4 (whose targets were not real built topologies).
#
#   bash scripts/run_real_data_round5.sh                 # resumable; skips finished cells
#   GRAPH_DIR=/path/to/total_graph_data bash scripts/run_real_data_round5.sh   # if graph_dir differs from legacy.yaml
#
# Steps: (0) build corpus v2 + stats  (A) ground truth + rule/search baselines + AR-GNN v3 (3 seeds)
#        (B) AR-GNN ablations: order (label / bfs / greedy), single-anchor loss, no planarity guard
#        (C) aggregate table  (D) regenerate thesis figures (F06/F06b/F07/F09 now use corpus v2 + planar_corridor decoder)
# Estimated on M4 Pro: (0) ~5 min for ~6 000 floors, (A) ~2 h (bestof16 dominates), (B) ~1.5 h, (D) ~5 min.
set -euo pipefail
cd "$(dirname "$0")/.."
GRAPH_DIR="${GRAPH_DIR:-}"
CORPUS="${CORPUS:-data/processed/legacy/stage2_corpus_v2.jsonl}"
S2_OUT=outputs/experiments/stage2_eval_r5; LIMIT="${S2_LIMIT:-600}"; SEEDS="${SEEDS:-0 1 2}"; DEVICE="${DEVICE:-auto}"

echo "== [0] build corpus v2 =="
if [[ ! -f "$CORPUS" ]]; then
  if [[ -n "$GRAPH_DIR" ]]; then python scripts/build_stage2_corpus.py --config configs/data/legacy.yaml --graph-dir "$GRAPH_DIR" --out "$CORPUS"
  else python scripts/build_stage2_corpus.py --config configs/data/legacy.yaml --out "$CORPUS"; fi
fi
[[ -f "$CORPUS" ]] || { echo "!! corpus not built: '$CORPUS'"; exit 2; }
echo "== corpus: $CORPUS  seeds: $SEEDS =="; cat "${CORPUS%.jsonl}.stats.json" || true

ev() { # cfg, seed, extra...
  local CFG=$1 SEED=$2; shift 2
  python scripts/evaluate_stage2.py --config "configs/stage2/$CFG.yaml" --corpus "$CORPUS" --limit "$LIMIT" --seed "$SEED" "$@" --override "eval_output_dir=$S2_OUT" || echo "!! $CFG seed $SEED failed"
}
tr() { python scripts/train_stage2.py --config "configs/stage2/$1.yaml" --corpus "$CORPUS" --override "stage2.generator.params.checkpoint=null" "stage2.generator.params.device=$DEVICE" || echo "!! train $1 failed"; }

echo "== [A] ground truth + baselines + AR-GNN v3 (3 seeds) =="
ev rule_baseline 0 --ground-truth
# thresholds calibrated on the ground-truth row (real built topologies are far denser than their skeletons)
python scripts/calibrate_stage2_thresholds.py --gt "$S2_OUT/ref_ground_truth/per_sample.csv" --out configs/stage2/thresholds_v2.local.yaml
TH="$(cat configs/stage2/thresholds_v2.local.yaml.override)"
ev() { local CFG=$1 SEED=$2; shift 2; python scripts/evaluate_stage2.py --config "configs/stage2/$CFG.yaml" --corpus "$CORPUS" --limit "$LIMIT" --seed "$SEED" "$@" --override "eval_output_dir=$S2_OUT" $TH || echo "!! $CFG seed $SEED failed"; }
ev rule_baseline 0 --ground-truth --force
[[ -f outputs/checkpoints/stage2/stage2_ar_gnn/ar_gnn.pt ]] && python - <<'PY' || true
import json, sys; m = json.load(open("outputs/checkpoints/stage2/stage2_ar_gnn/meta.json"))
sys.exit(0 if m.get("feat_version") == 3 else 1)
PY
if [[ $? -ne 0 || ! -f outputs/checkpoints/stage2/stage2_ar_gnn/ar_gnn.pt ]]; then tr ar_gnn; fi
for S in $SEEDS; do
  ev rule_baseline "$S"; ev search_baseline "$S"
  for CFG in ar_gnn ar_gnn_greedy ar_gnn_bestof16; do ev "$CFG" "$S" --override "stage2.generator.params.device=$DEVICE"; done
done

echo "== [B] AR-GNN v3 ablations (train once, eval 3 seeds) =="
for CFG in ar_gnn_bfs_order ar_gnn_greedy_order ar_gnn_single_label ar_gnn_no_planar; do
  [[ -f "configs/stage2/$CFG.yaml" ]] || continue
  [[ "$CFG" == ar_gnn_no_planar ]] || tr "$CFG"   # no_planar reuses the main checkpoint (inference-only switch)
  for S in $SEEDS; do ev "$CFG" "$S" --override "stage2.generator.params.device=$DEVICE"; done
done

echo "== [C] aggregate =="
python scripts/aggregate_stage2.py --root "$S2_OUT"

echo "== [D] thesis figures (uses corpus v2 test split + planar_corridor decoder) =="
mkdir -p data/results_snapshot/stage2 && cp -f "$S2_OUT"/table.md "$S2_OUT"/summary.csv data/results_snapshot/stage2/ 2>/dev/null || true
python scripts/make_thesis_report.py || echo "!! figures failed"
echo "== report back: $S2_OUT/table.md, ${CORPUS%.jsonl}.stats.json, outputs/checkpoints/stage2/*/meta.json, outputs/thesis/figures/{F06,F06b,F07,F09}*.png =="
