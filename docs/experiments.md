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
| deep_residual（Transformer+GIN 残差融合）| 0.681 ± 0.032 | 0.672 ± 0.019 | 0.430 | 0.304 | 0.677 |

**判读**：学习型 ranker 在不见评分的情况下，NDCG@10 达到上界的 92%（0.693 / 0.751），显著高于随机（0.522）；PairAcc 0.67–0.69。经典模型间差异在 seed 方差内，表格特征上的信号近似线性可分。
**第 3 轮（deep_residual，真实数据，已运行）**：残差融合深度 ranker NDCG@10 = 0.672 ± 0.019，与 ridge / lgbm_regressor / extra_trees（0.674–0.689）处于同一噪声带内，PairAcc 0.677 与经典模型持平；相比纯表格 MLP（0.575）提升 +0.10，方差从 0.047 降到 0.019。**结论：残差设计成功地把深度模型从"明显劣于经典"拉回到"与最强经典持平"，但在本数据上未超过 extra_trees**——这与 §2 的发现一致：候选拓扑结构对预测评分的边际信息有限，GNN 分支没有额外信号可挖。论文表述：深度模型的价值在于与经典模型**同一水平**下提供图结构编码接口（阶段二复用），而非排序精度提升。

### 1b. deep_residual 成分消融（真实数据，3 seeds，已运行）

| 变体 | NDCG@5 | NDCG@10 | Spearman | PairAcc |
|---|---|---|---|---|
| full（残差 + Transformer + GNN + 小样本技巧）| 0.672 ± 0.045 | 0.668 ± 0.024 | 0.306 | 0.678 |
| no_gnn | **0.704 ± 0.011** | **0.682 ± 0.005** | **0.311** | 0.677 |
| mlp_instead_of_transformer | 0.682 ± 0.003 | 0.676 ± 0.005 | 0.297 | 0.671 |
| no_smallsample_tricks | 0.657 ± 0.055 | 0.650 ± 0.028 | 0.288 | 0.669 |
| **no_residual（端到端）** | 0.631 ± 0.041 | **0.645 ± 0.024** | **0.259** | **0.652** |

**判读**（按效应从大到小）：(1) **残差结构最关键**：去掉后 NDCG@10 −0.023、Spearman −0.047、PairAcc −0.026，是唯一超出噪声带的下降——证实"以经典模型为主干、深度只学残差"是小样本下有效的设计原则；(2) **小样本技巧**（特征噪声 / 节点 dropout / 快照集成）去掉后 −0.018 且方差翻倍；(3) **GNN 分支去掉反而略升且方差最小**（0.682 ± 0.005）——与特征消融一致，候选图结构对评分无增量信息，GNN 在此任务上只增加方差；(4) Transformer→MLP 差异在噪声内。**最优实用配置是 `no_gnn`**（残差 + Transformer + 技巧），NDCG@10 0.682 与 extra_trees 0.689 持平且方差更小。

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

## 3. 阶段一 · 类型条件化质量模型 E[score | 条件, 布局类型] — 真实数据，5 seeds，已运行
模型：`TreeTypeRecommender`（ExtraTrees 回归 + bootstrap 置信区间），输入 = 10 个条件特征 + city_cluster + 布局类型 one-hot，输出 = 对每一种布局类型在**同一组条件下**的期望评分及 CI，并给出同 bucket 可比案例中的经验均值与类型占比作为证据。
评估（grouped split，test 商场）：
- `rmse / spearman`：带类型 vs 仅条件 → 类型带来的增量解释力；
- `per_cluster.tau_type_order`：每个城市簇内，模型给出的类型期望排序 vs 经验排序的 Kendall τ；
- `best_type_agreement_rate`：模型最优类型 = 该簇经验最优类型的比例；
- `policy_uplift`：按模型推荐类型 vs 按数据中实际类型的期望评分差。
**结果（test 520 楼层 / 119 商场，5 seeds）**

| 指标 | 值 |
|---|---|
| RMSE 带类型 / 仅条件 | 0.425 ± 0.003 / 0.427 ± 0.001 |
| Spearman 带类型 / 仅条件 | 0.573 ± 0.009 / 0.565 ± 0.002 |
| 簇内类型排序 Kendall τ（均值）| 0.40 ± 0.05（簇 1: 0.2–0.33，簇 2: 0.47–0.6，簇 3: 0.33–0.47）|
| best_type_agreement_rate | **1.00**（5/5 seeds × 3/3 簇）|
| policy_uplift（按推荐类型 vs 实际类型的均值评分差）| **+0.056 ± 0.026**（评分量程 2.9–4.9）|

按簇："按推荐类型建"与"未按推荐类型建"的均值评分：簇 1 4.65 vs 4.56（+0.09），簇 2 4.62 vs 4.25（**+0.37**），簇 3 4.32 vs 4.44（−0.13）。
**判读**：(1) 类型对评分的**增量解释力小但稳定为正**（RMSE −0.002，Spearman +0.008，5 seeds 中 4 个方向一致）——条件是一阶因素，类型是二阶效应，与既有分析一致；(2) 模型在每个簇内都正确识别出经验最优类型，且推荐类型的实际收益在簇 2（中等城市）最大；(3) **簇 3 policy uplift 为负**，说明簇 3 内"复杂集中型"并非普遍最优——这正是"最优类型随城市簇不同"的证据，也说明简单的"簇内最优类型"总结不够，应看**每簇的类型期望评分 ± CI 表**（第 3 轮脚本已输出 `summary.md`，含 `top1_separable_rate`）。下一轮需回传该表以判断簇 3 中哪些类型差异在 CI 之外。

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
| deep_residual（第 3 轮）| 0.925 | **0.633** | 3.57 | 4.88 | 0.626 |

条件→`type8` 可预测性（LightGBM, 4303/526）：准确率 31.9%，多数类 36.7%。
**判读**：各 ranker 的 `quality@K`（4.87–4.89）远高于随机（4.46）与多数类（4.43），说明检索到的确是**高质量的可比案例**；但 `type_precision@K` 与随机相当——策划条件本身不决定"业主实际选了哪种类型"（实际选型受土地形状、开发商偏好等本数据集未记录的因素影响）。这与 §3 的框架互补：**阶段一不是预测"会建什么类型"，而是回答"建哪种类型预期更好"，并据此检索该类型内的高质量可比原型**（`recommend_types → recommend_within_type`）。

## 5. 阶段二 · 真实 ShareGPT 语料（**最后 600 条留出**，seed 0）— 真实数据，Mac 第 3 轮（v2）已运行

### 5a. 大纲 5 指标 + 按标签的目标边匹配（Mac 运行）

| 生成器 | 5 指标全通过 | 节点偏差 | 密度偏差 | ASPL 偏差 | 连通分量 | 目标边召回 / 精度 | 推理时间 |
|---|---|---|---|---|---|---|---|
| **ground truth（语料真实拓扑，参照）** | 94.7% | 7.6% | 10.4% | 8.6% | 1.67 | 100 / 100 | – |
| rule_expander | 98.2% | 0.34% | 9.0% | 7.3% | 1.04 | 54.3 / 59.9 | 1 ms |
| search_expander(16) | **98.5%** | 0.34% | 9.1% | **2.2%** | 1.04 | 54.9 / 59.9 | 40 ms |
| ar_gnn v2（T=0.7）| 92.7% | 0.34% | 5.6% | 13.9% | **1.01** | 58.5 / 66.5 | 0.16 s |
| ar_gnn v2 greedy（T=0）| 70.7% | 0.34% | **4.2%** | 34.3% | 1.03 | **60.1 / 69.8** | 0.16 s |
| **ar_gnn v2 best-of-16** | 96.2% | 0.34% | 6.2% | 4.7% | **1.01** | 58.7 / 65.6 | 2.5 s |
| ar_gnn v1（BFS 顺序、单标签 CE，已弃用）| 94.3% | 0.34% | 4.9% | 12.8% | 1.02 | 55.3 / 64.4 | 0.12 s |

训练曲线（v2, 10 epoch, 72 s/epoch）：val anchor acc 0.374 → **0.411**（top-3 0.62 → 0.68），has2 acc 0.73 → 0.75，loss 5.46 → 4.01，仍在下降（v1: 0.16 → 0.17，平）。骨架边占目标边 52.8% = 任何保持骨架的生成器的召回下界。

### 5b. 无标签结构一致性（新增指标；沙箱用同一语料、同一 600 条留出、v2 8-epoch checkpoint 计算，Mac 重跑脚本已含）

语料中新节点的字母标签只有 65% 与"骨架节点数 + k"对齐，按标签的边匹配系统性低估任何生成器。新增三个不依赖新节点标签的指标：**attach recall/precision**（骨架节点 → 新分支的多重集合重叠：哪些骨架节点长出了几条分支）、**degree EMD**（与真实拓扑度分布的 1-D 距离）、**new–new 边比例**（新节点之间的边占新增边的比例，刻画"走廊式生长"）。

| 生成器 | attach recall | **attach precision** | degree EMD ↓ | new–new 比例（真实 0.338）|
|---|---|---|---|---|
| ground truth | 100 | 100 | 0 | 0.338 |
| rule_expander | 35.9 | 43.2 | 0.438 | 0.240 |
| search_expander(16) | 42.4 | 43.0 | 0.398 | 0.189 |
| **ar_gnn v2** | **51.3** | **79.7** | **0.242** | **0.362** |

**判读**
1. **AR-GNN v2 是第一个在结构指标上明显超过规则/搜索基线的生成器**：它把新分支接到正确骨架节点的精度是规则法的 1.8 倍（79.7 vs 43.2），度分布距离减半（0.24 vs 0.44），生长模式（new–new 比例 0.36）与真实语料（0.34）一致而规则法偏向"全接骨架"（0.19–0.24）。按标签的目标边召回/精度也一致地高 4–10 个百分点。
2. **大纲 5 指标不能区分好坏**：ground truth 自己只有 94.7% 通过率、1.67 个连通分量，低于所有生成器。大纲指标是合格线（"统计上像骨架"），不是目标函数；论文应把 ground truth 行放进表里说明这一点，并以 5b 的结构指标作为主要比较。
3. **采样温度是"结构正确 vs 大纲合格"的权衡**：greedy 精度最高（69.8）但 ASPL 偏差 34%（走廊过长，通过率 70%）；T=0.7 平衡；best-of-16 用大纲目标重排把 ASPL 压到 4.7%、通过率 96.2%，结构指标基本不损失。**推荐配置：ar_gnn best-of-16**（结构最优 + 大纲达标，2.5 s/样本仍在 60 s 限制内两个数量级以下）。
4. 与 search_expander 的对照公平：两者搜索预算相同（16 次采样 + 同一重排目标），差异只在 proposal 是学习的还是随机的——这就是学习型生成器贡献的干净度量。
5. 剩余空间：attach recall 51% 说明模型抓住了约一半真实分支的位置；val anchor acc 0.41 在 10 epoch 时仍在上升，更多 epoch / 更大模型（d=128）/ 束搜索是低成本的下一步；BFS-order 消融配置已备（`ar_gnn_bfs_order.yaml`，需单独训练）。

按布局（第 2 轮）：简单集中型通过率最低（77%，密度偏差 19.8%——星形骨架被"保持平均度"的期望密度公式惩罚）；其余 92–100%。

## 6. 第 3 轮实验清单（`bash scripts/run_real_data_phase4.sh`）
| 编号 | 内容 | 输出 |
|---|---|---|
| S1-A | 类型条件化质量模型，5 seeds | `outputs/experiments/real_r3/type_recommender/results.json` |
| S1-B | 模型比较 + deep_residual，3 seeds | `real_r3/model_comparison/table_test.md` |
| S1-C | 深度 ranker 成分消融（残差 / GNN / Transformer / 小样本技巧）| `real_r3/deep_ablation/table_test.md` |
| S1-D | 保真度协议：deep_residual vs extra_trees | `real_r3/fidelity/fidelity_summary.csv` |
| S2-A | 在语料去掉最后 600 条上训练 AR-GNN | `outputs/checkpoints/stage2/stage2_ar_gnn/` |
| S2-B | rule / search / ar_gnn / ar_gnn-bestof16 在同一 600 条留出上比较 | `outputs/experiments/stage2_eval_r3/table.md` |

第 3 轮状态：**全部已运行**（S1-A~D → §1/§1b/§3/§4；S2 v2 → §5a）。§5b 的无标签结构指标在沙箱计算，Mac 上 `SKIP_S1=1 bash scripts/run_real_data_phase4.sh` 重跑 S2-B（不重训，checkpoint 复用）即可得到 Mac 版本。

## 7. 合成数据（仅流程验证）
模型比较 / 消融 / 阶段二 / 保真度 / 类型推荐 / AR-GNN 训练与评估脚本均在合成数据上跑通；36 个测试通过；数值不具意义。

## Stage-2 corpus v2 (real CSV graph triplets) — supersedes rounds 1–4 for Stage 2

**Why.** `sharegpt_data.json` (rounds 1–4) turned out to be *LLM-generated* expansions, not real
built topologies. All Stage-2 numbers in rounds 1–4 (R09–R14, F06/F07) therefore measure agreement
with a synthetic target and are **superseded**; they are kept in `data/results_snapshot/stage2/` for
provenance only.

**Real corpus.** For every floor `{mall}_{k}` the graph export has `*_M_simplified.csv` (+ node
attributes) = corridor skeleton (Stage-1 prototype) and `*_M.csv` = complete corridor key-point
topology. `scripts/build_stage2_corpus.py` builds `data/processed/legacy/stage2_corpus_v2.jsonl`
with a mall-grouped, cluster-stratified split (same protocol as Stage 1). Verified on
`B000A0E928_1` (`tests/fixtures/graph_csv`): skeleton 25/28 → target 50/83; skeleton kept verbatim;
planar; connected; new nodes attach to 1–4 present nodes ({1: 6, 2: 9, 3: 9, 4: 1}).

**Model / decoder changes driven by the data.** AR-GNN v3 (iterative anchor/stop heads, up to 4
anchors, planarity guard; `FEAT_VERSION=3`, old checkpoints must be retrained); rule expander gains a
`bridge` op (new node closing a loop) and never adds skeleton–skeleton chords; geometry decoder
`planar_corridor` (Tutte embedding of the corridor 2-core, loop holes → atria capped at
`atrium_area_max`, branches perpendicular to the core, frontage-preserving shop split) replaces
`corridor_partition` as default. Outline thresholds are calibrated on the ground-truth row
(`scripts/calibrate_stage2_thresholds.py`, q=0.95) because real targets are much denser than skeletons.

**Run (Mac):** `bash scripts/run_real_data_round5.sh` (or `GRAPH_DIR=... bash ...`). Results → `outputs/experiments/stage2_eval_r5/table.md`.
