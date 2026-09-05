# 实验记录

> 规则：未在真实数据上运行的结果一律标注"待运行"。合成数据结果只用于验证流程。

## 真实数据审计（已运行，Mac，2026-09-05）
5395 行 / 1209 商场 / 5114 有图（266 无图）；97 列；`total_score` 5242 个非空，范围 2.9–4.9，中位 4.8，**仅 21 个取值、within-mall 方差 = 0**（商场级评分）；
8/10 条件特征在 mall 内恒定；Spearman(total_area, score)=0.48，图指标 |ρ|≈0.3，城市经济特征 |ρ|<0.18；15 个重复 floor_id（已在 adapter 去重）；不连通图 2.3%；三向 mall 泄漏 = 0。

## 阶段一模型比较
| 数据 | 状态 | 表格 |
|---|---|---|
| 真实 | **待运行** `bash scripts/run_real_data_phase2.sh` | outputs/experiments/real_model_comparison/table_test.md |
| 合成 (2 seeds, 流程验证) | 已运行 | weighted_rule(oracle) 0.999 · lgbm_lambdarank 0.984±0.014 · lgbm_regressor 0.975 · random_forest 0.975 · extra_trees 0.971 · ridge 0.961 · mlp 0.910 (NDCG@5) |

## 阶段一特征块消融（LambdaMART, 3 seeds）— 真实数据待运行

## 阶段二
| 生成器 | 语料 | 通过率 | 节点偏差 | 密度偏差 | ASPL 偏差 | 目标边召回 |
|---|---|---|---|---|---|---|
| rule_expander | 合成 120 | 96.7% | 0.0% | 9.2% | 16.1% | 58.1% |
| search_expander(16) | 合成 120 | 100% | 0.0% | 11.2% | 2.5% | 59.2% |
| rule / search | 真实 sharegpt 5632 | **待运行** | | | | |
