#!/usr/bin/env bash
# One-shot Phase 2/3 run on the REAL data (Mac). Produces everything to report back.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_data.py --config configs/data/legacy.yaml
python scripts/run_ablation.py --config configs/ablations/stage1_model_comparison.yaml --out-dir outputs/experiments/real_model_comparison
python scripts/run_ablation.py --config configs/ablations/stage1_feature_blocks.yaml --out-dir outputs/experiments/real_feature_ablation
python scripts/train_stage1.py --config configs/stage1/lgbm_lambdarank.yaml
SG="$(python -c "import yaml;print(yaml.safe_load(open('configs/data/legacy.yaml'))['sharegpt_json'])")"
if [ -f "$SG" ]; then
  python scripts/evaluate_stage2.py --config configs/stage2/rule_baseline.yaml --corpus "$SG" --limit 600
  python scripts/evaluate_stage2.py --config configs/stage2/search_baseline.yaml --corpus "$SG" --limit 600
else
  echo "sharegpt_json not found ($SG): set it in configs/data/legacy.yaml to evaluate stage 2 on the real corpus"
fi
python scripts/run_e2e.py --stage1-config configs/stage1/lgbm_lambdarank.yaml --checkpoint outputs/experiments/stage1/stage1_lgbm_lambdarank/seed_42/checkpoint \
  --condition data/samples/query_example.json --pick 1 --target-nodes 40 --target-shops 50 --shop-area 60 300 --out outputs/generated_layouts/e2e_real
echo "== report back: outputs/experiments/real_model_comparison/table_test.md, real_feature_ablation/table_test.md, stage2_eval/*/aggregate.json, generated_layouts/e2e_real/**/cand0.png =="
