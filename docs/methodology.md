# 方法说明（Phase 2–4）

## 阶段一：布局类型决策 → 质量感知的可比案例检索 → 用户选型
科学问题：**给定外部策划条件（城市簇、人口/经济、商业竞争、体量），哪种商场布局类型预期评分更高？** 并在该类型内检索高质量、条件相近的真实原型，作为设计/改造起点。

```
条件 c
→ (A) 类型决策 TreeTypeRecommender:  E[score | c, type_k] ± CI, k=1..6   →  类型排序 + 可比案例证据
→ (B) 用户选型 type*（默认取 #1；可改）
→ (C) 硬约束过滤(city_cluster, 面积档, layout=type*; 候选不足自动放宽并记录)
→ (D) kNN 召回 (标准化条件向量)  → Top-N
→ (E) ranker 重排 → 置信度 → 模板解释 + 反事实
```
服务接口：`PlanningService.recommend_types(c)` → `recommend_within_type(c, type*, top_k)`；CLI `run_e2e.py --use-top-type`。

**(A) 类型条件化质量模型**：ExtraTrees 回归 `f(c, cluster, onehot(type))`，bootstrap 重采样给出每类期望评分的置信区间；同时报告同 bucket 可比案例中该类型的经验均值与占比，保证"模型说的"和"数据里有的"同时呈现。评估见 `evaluate_type_recommender`（带类型 vs 仅条件的 RMSE/Spearman、簇内类型排序 τ、最优类型一致率、policy uplift）。

**(E) ranker 训练/评估协议（关键，与旧仓库不同）**
- 训练组：每个 train 楼层为 query，候选 = 同 bucket、**不同 mall** 的 train 楼层（同 mall 楼层 label 与全部 query 特征相同，会让模型学到"复制"）。
- 相关度 = 候选 `total_score` 组内 min-max；LambdaMART 用组内排名分位 grade（`grade_mode: rank`）。
- 评估：test 楼层为 query，候选 = 同 bucket 不同 mall 的 test 楼层；NDCG@K/P@K/R@K/Hit@K/MAP/MRR/Spearman/Kendall/PairAcc。上界 `quality_oracle`，下界 `random`。

**特征块（可消融）**：`condition`(10 query 列, 重尾 log1p) · `prototype_metrics`(L1/L2) · `graph_metrics`(重算 11 列) · `extra_metrics`(真实表 18 列) · `match`(|q−c|, q−c, log 面积比, 同 cluster)。**`total_score` 绝不作为特征。**

**Ranker 注册表**：`random`, `quality_oracle`, `weighted_rule`, `ridge`, `random_forest`, `extra_trees`（默认）, `mlp`, `lgbm_regressor`, `lgbm_lambdarank`, **`deep_residual`**。

### deep_residual：Transformer + GNN 残差融合（小样本设计）
```
s(q, p) = w · s_tab(q, p)  +  α · s_deep(q, p),   α 初始 0.1、可学习
s_tab   : 已训练好的经典 ranker（默认 extra_trees）—— 强先验，深度部分只学残差
s_deep  : [ 表格特征 MLP 编码 ‖ 原型图 GIN 读出(2 层, 节点特征 = log度/聚类系数/介数/坐标) ]
          → set-Transformer(1 层, 4 头) 在同一 query 的候选集合内做上下文交互 → 线性打分
损失    : listwise softmax 交叉熵（组内相对排序）
小样本技巧 : 残差初始化(α 小 → 起点即经典模型) · 特征高斯噪声 · 节点 dropout · 权重衰减 ·
             早停 · snapshot ensembling(K=3) · 纯 PyTorch GIN，无 PyG 依赖，MPS/CUDA/CPU 自动
```
消融（`configs/ablations/stage1_deep_ablation.yaml`）：去残差(端到端)、去 GNN、Transformer→MLP、去小样本技巧。

### ⚠️ 关于 weighted_rule / quality_oracle 的诚实说明
两者直接读取候选 `total_score`（数据库属性），而评估相关度就是 `total_score`，因此是 **上界参照**而非公平基线。学习型 ranker 的价值在于：给定新项目条件、**没有评分**时预测哪些原型质量高且匹配。真实数据上学习型 ranker NDCG@10 达到上界的 92%，显著高于随机。

## 阶段二：原型保持的可控拓扑扩展 + 计算几何 + 修复
```
generator: rule_expander | search_expander(16 采样取大纲指标最优) | ar_gnn (自回归 GNN) | ar_gnn best-of-16
→ geometry_decoder: corridor_partition (骨架坐标固定+弹簧布局放新节点→缩放入边界→走廊缓冲→中庭→临街带切店铺/纵深切主力店)
→ repairer: basic (连通修复、越界拉回、小店合并、不可达删除；全部记录在 diagnostics.repairs)
→ evaluator: topology_spec(大纲 5 指标 + 目标边召回/精度) + geometry(边界内比例/重叠率/无效多边形/可达率/面积分布/约束满足率)
→ 候选按 (全部硬检查通过, 约束满足率, 节点偏差) 排序
```
### ar_gnn：自回归 GNN 扩展器
- **状态**：当前部分图（骨架 + 已加节点）；节点特征 = [是否骨架, log 度, 归一化度, 时间步比例, 已加节点比例] ‖ 布局类型 one-hot；全局上下文 = [t/T, N_target 归一化] ‖ 类型 one-hot。
- **一步**：GIN(3 层) 编码 → 三个头：`anchor`（新节点接到哪个已有节点，softmax over nodes，有限 mask −30）、`has2`（是否再加第二条边）、`second`（第二条边端点，条件于 anchor）。
- **训练**：teacher forcing——真实扩展按 BFS 距骨架的规范顺序拆成 T 步；anchor 用 label-smoothing CE，second 用普通 CE，has2 用 BCE；早停看 val anchor 准确率；snapshot ensemble。
- **推理**：温度采样逐步加 N_target − N_skeleton 个节点；**骨架节点与边始终不变**（原型保持是结构性保证而非软约束）；`best_of>1` 时按 节点/密度/ASPL 偏差 + 连通惩罚 挑最优。
- **可比性**：与 rule/search 使用同一评估器、同一 600 条留出（训练集不含）。

合成语料上 rule 通过率 96.7%，search 100%，ar_gnn 100%（流程验证）。真实数据见 `docs/experiments.md` §5。
