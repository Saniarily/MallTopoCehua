#!/usr/bin/env bash
# Phase 2/3 experiments on the REAL data (Mac). Round 2: reference lines, fixed LambdaMART, fidelity protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_data.py --config configs/data/legacy.yaml
# A) quality protocol: all rankers incl. random lower bound & quality-oracle upper bound (3 seeds)
python scripts/run_ablation.py --config configs/ablations/stage1_model_comparison.yaml --out-dir outputs/experiments/real_model_comparison_r2
# B) prototype-fidelity protocol (label-free) + layout predictability
python scripts/evaluate_fidelity.py --config configs/stage1/ridge.yaml \
  --configs configs/stage1/lgbm_lambdarank.yaml configs/stage1/extra_trees.yaml configs/stage1/rule_knn.yaml configs/stage1/quality_oracle.yaml \
  --out outputs/experiments/real_fidelity
echo "== report back: outputs/experiments/real_model_comparison_r2/table_test.md, real_fidelity/fidelity_summary.csv, real_fidelity/layout_predictability.json =="
