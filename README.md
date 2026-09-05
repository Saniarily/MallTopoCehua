# MallTopoCehua — 商场空间智能策划辅助决策系统

以**拓扑原型**为中间态的两阶段框架：阶段一检索/排序/解释拓扑原型；阶段二在场地约束下可控扩展拓扑并生成平面布局草案。
本仓库从零实现，旧仓库 [MallTopoRanker](https://github.com/Saniarily/MallTopoRanker) 仅作只读参考（分析见 `docs/legacy_repo_analysis.md`）。

## 安装（Mac M4 Pro，conda 环境 `mallranker` / Python 3.11）
```bash
conda activate mallranker
pip install -e ".[dev]"            # 核心依赖；可选: ".[boosting]" ".[deep]" ".[app]" ".[tracking]"
pytest -v                          # 13 tests（含合成数据 smoke test）
```

## 经过验证的命令
```bash
# 旧仓库只读审计 → outputs/reports/legacy_repo_audit.{md,json}
python scripts/audit_legacy_repo.py --legacy-repo /Users/saniarily/Desktop/Coding_Mac/Mall_Topo/MallTopoRanker
# 数据准备（合成数据；真实数据改用 configs/data/legacy.yaml，路径可用 --override 覆盖）
python scripts/prepare_data.py --config configs/data/synthetic.yaml
python scripts/prepare_data.py --config configs/data/legacy.yaml --override processed_dir=data/processed/legacy
# 阶段一训练（默认读真实数据 data/processed/legacy；合成数据加 --override data.processed_dir=data/processed/synthetic）
# 模型切换只改配置：rule_knn | ridge | random_forest | extra_trees | mlp | lgbm_regressor | lgbm_lambdarank
python scripts/train_stage1.py --config configs/stage1/lgbm_lambdarank.yaml            # 写 val/test 指标 + checkpoint + run.json
python scripts/evaluate_stage1.py --config configs/stage1/lgbm_lambdarank.yaml \
    --checkpoint outputs/experiments/stage1/stage1_lgbm_lambdarank/seed_42/checkpoint
# 多 seed 模型比较 / 特征块消融 → summary_*.csv, table_*.md, ndcg10_*.png
python scripts/run_ablation.py --config configs/ablations/stage1_model_comparison.yaml
python scripts/run_ablation.py --config configs/ablations/stage1_feature_blocks.yaml
python scripts/compare_experiments.py --root outputs/experiments
# 阶段二：生成 + 大纲 5 指标 + 几何检查 + JSON/GeoJSON/SVG/PNG；在 skeleton→topology 语料上批量评估
python scripts/generate_stage2.py --config configs/stage2/search_baseline.yaml --target-nodes 40 --target-shops 50 --shop-area 60 300
python scripts/evaluate_stage2.py --config configs/stage2/search_baseline.yaml --corpus /path/to/sharegpt_data.json --limit 600
# 端到端：策划条件 JSON → Top-K → 选择第 k 个原型 → 场地约束生成 → 导出
python scripts/run_e2e.py --condition data/samples/query_example.json --pick 1 --width 180 --height 120 \
    --target-nodes 40 --target-shops 50 --shop-area 60 300
# 真实数据一键跑完 Phase 2/3 全部实验（Mac）
bash scripts/run_real_data_phase2.sh
```

## 目录
`configs/` 数据/阶段/消融配置 · `src/mall_space_planner/` {schemas, data, features, topology, geometry, stage1, stage2, evaluation, registry, utils} ·
`scripts/` CLI · `tests/` unit+smoke · `docs/` 审计/Schema/架构 · `data/samples/` 合成样例（自动生成）。

## 状态（Phase 0–3 完成）
- ✅ 审计 / Schema / registry / 适配器（含真实 97 列表的 18 个额外拓扑列与 `type8` 布局类型）/ 无泄漏划分
- ✅ 阶段一：硬约束过滤 + kNN 召回 + 7 种 ranker（含 **LambdaMART**、MLP）+ 模板解释 + 反事实 + 多 seed 消融/对比脚本
- ✅ 阶段二：rule / **search** 扩展器 + 走廊缓冲/中庭/临街店铺/主力店分区 + 修复器 + 拓扑&几何双评估 + JSON/GeoJSON/SVG/PNG
- ✅ 端到端 `run_e2e.py` 与 UI 无关的 `PlanningService`；21 个测试通过
- ⏳ 待运行：真实数据全部实验（`bash scripts/run_real_data_phase2.sh`）
- ⏳ 待实现（Phase 4–5）：GNN/双塔残差融合、学习型图编辑生成器、学习型校准器、MLflow、Streamlit Viewer Hub、FastAPI
