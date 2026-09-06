# MallTopoCehua — 商场空间智能策划辅助决策系统

以**拓扑原型**为中间态的两阶段框架：阶段一检索/排序/解释拓扑原型；阶段二在场地约束下可控扩展拓扑并生成平面布局草案。
本仓库从零实现，旧仓库 [MallTopoRanker](https://github.com/Saniarily/MallTopoRanker) 仅作只读参考（分析见 `docs/legacy_repo_analysis.md`）。

## 安装（Mac M4 Pro，conda 环境 `mallranker` / Python 3.11）
```bash
conda activate mallranker
pip install -e ".[dev]"            # 核心依赖；可选: ".[boosting]" ".[deep]" ".[app]" ".[tracking]"
pytest -v                          # 36 tests（含合成数据 smoke test；深度模型测试需 .[deep]）
```

## 经过验证的命令
```bash
# 旧仓库只读审计 → outputs/reports/legacy_repo_audit.{md,json}
python scripts/audit_legacy_repo.py --legacy-repo /Users/saniarily/Desktop/Coding_Mac/Mall_Topo/MallTopoRanker
# 数据准备（合成数据；真实数据改用 configs/data/legacy.yaml，路径可用 --override 覆盖）
python scripts/prepare_data.py --config configs/data/synthetic.yaml
python scripts/prepare_data.py --config configs/data/legacy.yaml --override processed_dir=data/processed/legacy
# 阶段一训练（默认读真实数据 data/processed/legacy；合成数据加 --override data.processed_dir=data/processed/synthetic）
# 模型切换只改配置：rule_knn | ridge | random_forest | extra_trees(默认) | mlp | lgbm_regressor | lgbm_lambdarank | deep_residual(Transformer+GNN 残差融合)
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
# 原型保真度协议（label-free）+ 条件→布局类型可预测性
python scripts/evaluate_fidelity.py --config configs/stage1/ridge.yaml --configs configs/stage1/lgbm_lambdarank.yaml
# 布局类型决策：E[score | 条件, 类型]（多 seed 评估）；端到端时用 --use-top-type 只在推荐类型内检索
python scripts/evaluate_type_recommender.py --config configs/stage1/base.yaml --seeds 42 43 44
python scripts/run_e2e.py --condition data/samples/query_example.json --use-top-type --pick 1 --width 180 --height 120
# 阶段二 AR-GNN：训练（自动留出语料最后 600 条）→ 与 rule/search 在同一留出集比较
python scripts/train_stage2.py --config configs/stage2/ar_gnn.yaml --corpus /path/to/sharegpt_data.json --override stage2.generator.params.checkpoint=null
python scripts/evaluate_stage2.py --config configs/stage2/ar_gnn_bestof16.yaml --corpus /path/to/sharegpt_data.json --limit 600
# 真实数据一键脚本（Mac）：第 1–2 轮 / 第 3 轮（Phase 4）/ 第 4 轮（阶段二消融与多 seed）
bash scripts/run_real_data_phase2.sh
bash scripts/run_real_data_phase4.sh
bash scripts/run_real_data_round4.sh
# 论文全部图表（PNG/PDF/SVG，中文字体自动探测；样式与中文标签在 configs/thesis/style.yaml）
python scripts/make_thesis_report.py                 # → outputs/thesis/figures/ ；报告正文见 docs/thesis/thesis_report.md
python scripts/make_thesis_report.py --only R05 R09  # 只重生成指定图
```

## 目录
`configs/` 数据/阶段/消融配置 · `src/mall_space_planner/` {schemas, data, features, topology, geometry, stage1, stage2, evaluation, registry, utils} ·
`scripts/` CLI · `tests/` unit+smoke · `docs/` 审计/Schema/架构 · `data/samples/` 合成样例（自动生成）。

## 机器相关路径
复制 `configs/data/legacy.local.yaml.example` → `legacy.local.yaml`（git 忽略）填写本机路径；任何 `<cfg>.local.yaml` 都会自动覆盖同名配置。

## 状态（Phase 0–4 代码完成；Phase 4 真实数据待运行）
- ✅ 审计 / Schema / registry / 适配器（含真实 97 列表的 18 个额外拓扑列与 `type8` 布局类型）/ 无泄漏划分
- ✅ 阶段一：**类型条件化质量模型** E[score | 条件, 类型]（`recommend_types → recommend_within_type`）+ 硬约束过滤 + kNN 召回 + 10 种 ranker（经典 / LTR / 上下界参照 / **deep_residual** Transformer+GIN 残差融合）+ 解释 + 反事实 + 多 seed 消融
- ✅ 阶段二：rule / search / **ar_gnn**（自回归 GNN，原型结构性保持）扩展器 + 几何解码 + 修复器 + 拓扑&几何双评估 + JSON/GeoJSON/SVG/PNG
- ✅ 原型保真度协议、真实数据第 1–2 轮结果（`docs/experiments.md`）；36 个测试通过
- ✅ 第 3 轮真实实验完成：deep_residual 与经典持平（残差结构是关键成分）；类型模型 best-type 一致率 100%、policy uplift +0.056；**AR-GNN v2 在结构指标上明显超过规则/搜索**（attach precision 79.7 vs 43.2）——见 `docs/experiments.md`
- ⏳ 待实现（Phase 5）：Streamlit Viewer Hub（数据浏览 / 实验看板 / 策划工作台）、FastAPI、学习型校准器、MLflow（可选）
- ✅ 第 4 轮：阶段二 3 seeds + AR-GNN 三项消融 + 大模型；**全部实验完成**。论文报告：`docs/thesis/thesis_report.md`（21 张图 `docs/thesis/figures/`）
详见 `docs/methodology.md`、`docs/innovation_points.md`、`docs/experiments.md`。
