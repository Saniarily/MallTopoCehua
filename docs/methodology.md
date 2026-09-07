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
### ar_gnn：自回归 GNN 扩展器（v2）
- **状态**：当前部分图（骨架 + 已加节点）；节点特征 = [log 度, 聚类系数, 是否骨架, 是否新节点, t/N_target, 节点年龄, 是否上一步刚加, 到上一新节点的 BFS 距离, 骨架内度, 新邻居数, 叶子, 相对度, 孤立] ‖ 4 阶随机游走结构编码(RWSE) ‖ 布局 one-hot；全局上下文 = [t/T, 平均度, log N_target] ‖ 类型 one-hot；每层 GIN 与每个头都拼接全图 mean-readout。
- **一步**：GIN(3 层) 编码 → 三个头：`anchor`（新节点接到哪个已有节点）、`has2`（是否再加第二条边）、`second`（第二条边端点，条件于 anchor，anchor 位置 mask −30）。
- **训练**：teacher forcing 按**语料自身的标签顺序**（字母标签 = 创建顺序；邻居尚未出现的节点延后），BFS 顺序保留为消融；anchor / second 用**集合似然** −log Σ_{合法端点} p（31% 的步骤有多个合法锚点）；has2 用 BCE；64 步一批，OneCycle 学习率；早停看 val anchor 准确率；snapshot ensemble。
- **推理**：温度采样逐步加 N_target − N_skeleton 个节点；**骨架节点与边始终不变**（原型保持是结构性保证而非软约束）；`best_of>1` 时按 节点/密度/ASPL 偏差 + 连通惩罚 挑最优。
- **可比性**：与 rule/search 使用同一评估器、同一 600 条留出（训练集不含）。

合成语料上 rule 通过率 96.7%，search 100%，ar_gnn 100%（流程验证）。真实数据见 `docs/experiments.md` §5。

## 阶段三：走廊关键点网络的轮廓自适应 + 一键成廊（不做商铺分区）
输入：阶段二得到的**完整 M 关键点网络**（只用拓扑，不用坐标）+ 一个外轮廓（数据库中相近面积的真实楼层：`*_total.csv` 多边形并集 / 轮廓 mask png；或网页端手绘多边形）。输出：关键点坐标（米）+ 完整走廊几何（主/次走廊多边形、出入口、中庭）。剩余空间留给设计师。

```
outline (px→m, flip_y)  →  inset = outline ⊖ 店铺纵深 d   →  medial axis (skimage skeletonize, 去毛刺)
→ CorridorFitter.fit(topology, outline, seed):
   1) planar_corridor_embedding 取抽象平面图：外环(最长面) / 内核(2-core) / 支路 / 叶
   2) 初始化：外环节点按等弧长钉在 inset 边界上(n_offsets 个相位×2 方向，靠近拐角吸附拐角)；内核 Tutte 重心；支路 BFS 沿中轴向外
   3) 松弛(iters 步, 退火步长)：外环→边界吸引；支路→中轴吸引；边方向→主墙向正交吸附；非邻接点排斥(min_spacing)；边-点净距；度 2 直通；
      每步投影回 inset；会产生交叉的移动按 1, ½, ¼, ⅛ 线搜索缩步，仍交叉则放弃（平面性硬保证）
   4) 拐角吸附；n_restarts 个候选按综合分挑最优（seed>0 时从 2n 池中抽样，实现"再试一次"）
→ render_corridors: 边介数 → 主/次；宽度由 corridor_ratio(≈0.18) 预算反推并裁到 [3,8] m；交叉口垫片；
   出入口 = 悬挂端/外环点向最近立面延伸(≥2 个，间距≥30 m)；被外环围出的洞(面积 ≤ atrium_area_max) = 中庭
→ evaluate_fit: inside_ratio / crossings / ortho_deviation / served_area / corridor_ratio / n_entrances
   + 与真实同层 M CenterPoint 对比：procrustes_rmse(带标签)、chamfer(无标签)、各自的随机基线、外环到立面距离(拟合 vs 真实)
```

生成规则（从真实平面归纳，见 `stage3/fit.py` 顶部注释）：R1 外环走廊沿立面内侧一个店铺纵深；R2 环内的洞是中庭，不是店铺；R3 支路垂直于所在环/主墙向；R4 走廊尽量顺主墙向、拐角处转折；R5 非邻接关键点 ≥ min_spacing（随 inset 面积/节点数自适应）；R6 拓扑不改、平面性不破；R7 走廊面积占比 ≈ 真实分布 0.12–0.25。

样例（B000A0E928_1，50 个 M 点，走廊来自 `_total.csv`）：crossings 0、inside 1.0、ortho 8°、corridor 28%、2 出入口 2 中庭；chamfer 明显优于随机（测试 `tests/unit/test_stage3_corridor.py`）。**Procrustes 在细长楼层上被长轴主导（随机放置也只有 ≈0.2×对角线），报告时以 chamfer + 随机基线为主。** 全库评估需要 Mac 上的 `*_total.csv`（`scripts/preview_stage3.py` 单层；批量脚本待补）。
