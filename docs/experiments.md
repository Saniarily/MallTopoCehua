# 实验记录

> 规则：未在真实数据上运行的结果一律标注 **待运行**。合成数据结果只用于验证流程。所有真实结果由用户在 Mac 上运行 `scripts/run_real_data_phase2.sh`（第 1–2 轮）与 `scripts/run_real_data_phase4.sh`（第 3 轮）得到。

## 0. 真实数据审计（Mac，2026-09-05）
5380 行（去重 15 个 floor_id）/ 1209 商场 / 5114 有图（266 无图）；97 列；`total_score` 范围 2.9–4.9，中位 4.8，**仅 21 个取值、within-mall 方差 = 0**（商场级评分）；
8/10 条件特征在 mall 内恒定；Spearman(total_area, score)=0.48，图指标 |ρ|≈0.3，城市经济 |ρ|<0.18；不连通图 2.3%；三向 mall 泄漏 = 0；部分数值列含 `#DIV/0!`（已转 NaN 并计数）。

## 1. 阶段一 · 质量排序协议 — 真实数据，3 seeds，第 2 轮已运行
协议：test 商场楼层为 query，候选 = 同 bucket、不同 mall 的 test 楼层，相关度 = 候选 `total_score` 组内 min-max。上界 = 直接按 `total_score` 排序（模型见不到该列），下界 = 随机排序。

| ranker | NDCG@5 | NDCG@10 | MAP | Spearman | PairAcc |
|---|---|---|---|---|---|
| quality_oracle（上界，读 label）| 0.751 | 0.751 | 0.808 | 0.751 | 0.963 |
| weighted_rule（读 label + 相似度，参照）| 0.726 | 0.727 | 0.642 | 0.642 | 0.927 |
| **extra_trees**（默认）| **0.714 ± 0.009** | **0.693 ± 0.007** | 0.456 | 0.295 | 0.673 |
| ridge | 0.699 ± 0.011 | 0.674 ± 0.004 | 0.463 | 0.333 | 0.693 |
| lgbm_regressor | 0.688 ± 0.021 | 0.677 ± 0.016 | 0.452 | 0.320 | 0.680 |
| lgbm_lambdarank（value-grade）| 0.667 ± 0.029 | 0.662 ± 0.019 | 0.377 | 0.283 | 0.651 |
| lgbm_lambdarank（rank-grade）| 0.656 ± 0.006 | 0.643 ± 0.012 | 0.426 | 0.339 | 0.679 |
| random_forest | 0.626 ± 0.054 | 0.627 ± 0.040 | 0.403 | 0.228 | 0.627 |
| mlp（纯表格 MLP）| 0.558 ± 0.088 | 0.575 ± 0.047 | 0.326 | 0.190 | 0.609 |
| random（下界）| 0.521 ± 0.004 | 0.522 ± 0.003 | 0.244 | −0.002 | 0.481 |
| **deep_residual（Transformer+GIN 残差融合）** | 待运行（第 3 轮）| | | | |

**判读**：学习型 ranker 在不见评分的情况下，NDCG@10 达到上界的 92%（0.693 / 0.751），显著高于随机（0.522）；PairAcc 0.67–0.69。经典模型间差异在 seed 方差内，表格特征上的信号近似线性可分。表格 MLP 表现最差、方差最大——这说明**小样本下端到端深度模型需要专门设计**（残差融合、集成、噪声正则），而非深度模型本身无用；第 3 轮的 `deep_residual` 即为此设计。

## 2. 阶段一 · 特征块消融（LambdaMART, 3 seeds）— 真实数据，已运行

| variant | NDCG@10 | Spearman | PairAcc |
|---|---|---|---|
| full | 0.662 ± 0.019 | 0.283 | 0.651 |
| no_graph_metrics | 0.675 ± 0.011 | 0.292 | 0.659 |
| no_legacy_metrics(L1/L2) | 0.682 ± 0.033 | 0.306 | 0.669 |
| no_extra_metrics | 0.675 ± 0.020 | 0.232 | 0.621 |
| no_condition | 0.665 ± 0.006 | 0.276 | 0.643 |
| **no_match** | 0.662 ± 0.012 | **0.221** | **0.612** |
| no_retriever | 0.662 ± 0.019 | 0.283 | 0.651 |

**判读**：唯一稳定信号是 **条件–候选匹配特征**（去掉后 PairAcc/Spearman 明显下降）；候选自身拓扑指标对"预测评分"贡献很小。与既有数据分析一致：**评分主要由城市/场地条件解释，布局类型是二阶效应但存在因果影响、且最优类型随城市簇不同**——因此阶段一的核心科学问题不是"拓扑指标预测评分"，而是 **条件效应问题**：给定外部策划条件，哪种布局类型的期望评分更高（§3）。`no_retriever` 与 full 相同是协议原因（评估时候选集固定）。

## 3. 阶段一 · 类型条件化质量模型 E[score | 条件, 布局类型] — **待运行（第 3 轮 S1-A）**
模型：`TreeTypeRecommender`（ExtraTrees 回归 + bootstrap 置信区间），输入 = 10 个条件特征 + city_cluster + 布局类型 one-hot，输出 = 对每一种布局类型在**同一组条件下**的期望评分及 CI，并给出同 bucket 可比案例中的经验均值与类型占比作为证据。
评估（grouped split，test 商场）：
- `rmse / spearman`：带类型 vs 仅条件 → 类型带来的增量解释力；
- `per_cluster.tau_type_order`：每个城市簇内，模型给出的类型期望排序 vs 经验排序的 Kendall τ；
- `best_type_agreement_rate`：模型最优类型 = 该簇经验最优类型的比例；
- `policy_uplift`：按模型推荐类型 vs 按数据中实际类型的期望评分差。
合成数据 smoke：`rmse type/cond=10.06/12.79, tau_type_order=1.00, best_type_agree=1.00`（无意义，仅流程）。

## 4. 阶段一 · 原型保真度协议（label-free）— 真实数据，第 2 轮已运行

| 方法 | type_hit@5 | type_precision@5 | metric_dist@5 | quality@5 | type_precision@10 |
|---|---|---|---|---|---|
| ref_oracle（知道真实类型）| 1.000 | 1.000 | 3.53 | 4.31 | 1.000 |
| ref_majority（永远推最常见类型）| 0.767 | 0.767 | 3.85 | 4.43 | 0.767 |
| ref_random | 0.942 | 0.598 | 3.70 | 4.46 | 0.606 |
| rule_knn | 0.933 | 0.618 | **3.39** | 4.87 | **0.643** |
| extra_trees | 0.917 | 0.627 | 3.54 | **4.89** | 0.632 |
| lgbm_lambdarank | 0.950 | 0.620 | 3.45 | 4.88 | 0.609 |
| ridge | 0.942 | 0.603 | 3.94 | 4.75 | 0.621 |
| quality_oracle | 0.925 | 0.603 | 3.50 | 4.90 | 0.616 |

条件→`type8` 可预测性（LightGBM, 4303/526）：准确率 31.9%，多数类 36.7%。
**判读**：各 ranker 的 `quality@K`（4.87–4.89）远高于随机（4.46）与多数类（4.43），说明检索到的确是**高质量的可比案例**；但 `type_precision@K` 与随机相当——策划条件本身不决定"业主实际选了哪种类型"（实际选型受土地形状、开发商偏好等本数据集未记录的因素影响）。这与 §3 的框架互补：**阶段一不是预测"会建什么类型"，而是回答"建哪种类型预期更好"，并据此检索该类型内的高质量可比原型**（`recommend_types → recommend_within_type`）。

## 5. 阶段二 · 真实 ShareGPT 语料（最后 600 条留出，seed 0）

| 生成器 | 5 指标全通过 | 节点偏差 | 边准确率 | 密度偏差 | ASPL 偏差 | 推理时间 | 目标边召回 / 精度 |
|---|---|---|---|---|---|---|---|
| rule_expander（第 2 轮）| 95.5% | 1.35% | 100% | 8.2% | 6.4% | 0.6 ms | 60.3% / 71.8% |
| search_expander(16)（第 2 轮）| **95.8%** | 1.35% | 100% | 8.3% | **2.5%** | 22 ms | 61.0% / 71.4% |
| **ar_gnn**（自回归 GNN）| 待运行（第 3 轮）| | | | | | |
| **ar_gnn best-of-16** | 待运行（第 3 轮）| | | | | | |

> 注：第 2 轮 rule/search 是在语料**前** 600 条上评估；第 3 轮统一改为**最后** 600 条留出（AR-GNN 训练集不含），四种方法会在 `run_real_data_phase4.sh` 中在同一留出集重跑，以上两行届时以第 3 轮数字为准。

按布局：简单集中型通过率最低（77%，密度偏差 19.8%——星形骨架被"保持平均度"的期望密度公式惩罚）；其余 92–100%。
**判读**：规则/搜索基线满足大纲全部合格判据，但**目标边召回 ≈ 60% ≈ 骨架边占比**，即新增连接基本不命中真实连接——统计指标可满足，结构模式未学到。AR-GNN 的目标正是在保持大纲指标的同时提高目标边召回/精度（学到数据中的连接模式）。
合成数据 smoke（40 条留出，4 epoch）：ar_gnn 通过率 100%，目标边召回 62.2%（无意义，仅流程）。

## 6. 第 3 轮实验清单（`bash scripts/run_real_data_phase4.sh`）
| 编号 | 内容 | 输出 |
|---|---|---|
| S1-A | 类型条件化质量模型，5 seeds | `outputs/experiments/real_r3/type_recommender/results.json` |
| S1-B | 模型比较 + deep_residual，3 seeds | `real_r3/model_comparison/table_test.md` |
| S1-C | 深度 ranker 成分消融（残差 / GNN / Transformer / 小样本技巧）| `real_r3/deep_ablation/table_test.md` |
| S1-D | 保真度协议：deep_residual vs extra_trees | `real_r3/fidelity/fidelity_summary.csv` |
| S2-A | 在语料去掉最后 600 条上训练 AR-GNN | `outputs/checkpoints/stage2/stage2_ar_gnn/` |
| S2-B | rule / search / ar_gnn / ar_gnn-bestof16 在同一 600 条留出上比较 | `outputs/experiments/stage2_eval_r3/table.md` |

## 7. 合成数据（仅流程验证）
模型比较 / 消融 / 阶段二 / 保真度 / 类型推荐 / AR-GNN 训练与评估脚本均在合成数据上跑通；36 个测试通过；数值不具意义。
