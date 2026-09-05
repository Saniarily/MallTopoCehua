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
# 阶段一训练 / 评估（切换模型只改配置：rule_knn | random_forest | extra_trees | ridge）
python scripts/train_stage1.py --config configs/stage1/random_forest.yaml
python scripts/evaluate_stage1.py --config configs/stage1/random_forest.yaml \
    --checkpoint outputs/experiments/stage1/stage1_random_forest/checkpoint
# 阶段二规则基线生成 + 大纲 5 指标评估 + JSON/GeoJSON/SVG 导出
python scripts/generate_stage2.py --config configs/stage2/rule_baseline.yaml --target-nodes 40
```
真实数据时把 `configs/stage1/base.yaml` 的 `data.processed_dir` 改为 `data/processed/legacy`（或 `--override data.processed_dir=...`）。

## 目录
`configs/` 数据/阶段/消融配置 · `src/mall_space_planner/` {schemas, data, features, topology, geometry, stage1, stage2, evaluation, registry, utils} ·
`scripts/` CLI · `tests/` unit+smoke · `docs/` 审计/Schema/架构 · `data/samples/` 合成样例（自动生成）。

## 状态（Phase 0–1 完成）
- ✅ 旧仓库审计、数据审计、Schema、registry/factory、legacy/synthetic/ShareGPT 适配器、按 mall 分组无泄漏划分
- ✅ 阶段一基线：硬约束过滤 + kNN 召回 + {规则加权, RandomForest, ExtraTrees, Ridge} 重排 + 模板解释 + 反事实 + 置信度
- ✅ 阶段二基线：规则扩展器（骨架边 100 % 保留）+ 边界内嵌入 + 大纲 5 指标评估器 + 3 种导出
- ⏳ 待运行：真实数据上的全部指标（本沙箱无真实 CSV/图目录；合成数据结果不代表实验结论）
- ⏳ 待实现：LTR/LightGBM、MLP/GNN/双塔、店铺分区与修复器、消融/对比脚本、MLflow、Streamlit Viewer Hub、FastAPI
