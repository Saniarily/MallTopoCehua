# 方法说明（Phase 2–3 基线）

## 阶段一：召回 + 重排
```
硬约束过滤(city_cluster, 面积档, 可选 layout 偏好; 候选不足自动放宽并记录)
→ kNN 召回 (标准化条件向量, 可加权)  → Top-N
→ ranker 重排  → 置信度(分数 z-score→sigmoid, 待学习型校准器替换) → 模板解释 + 反事实
```
**训练/评估协议（关键，与旧仓库不同）**
- 训练组：每个 train 楼层为 query，候选 = 同 bucket、**不同 mall** 的 train 楼层（同 mall 楼层的 label 与全部 query 特征完全相同，会让模型学到"复制"）。
- 相关度 = 候选 `total_score` 在组内 min-max；LambdaMART 用 5 级整数 grade。
- 评估：test 楼层为 query，候选 = 同 bucket 不同 mall 的 test 楼层；NDCG@K/P@K/R@K/Hit@K/MAP/MRR/Spearman/Kendall/PairAcc。

**特征块（可消融）**：`condition`(10 query 列, 重尾 log1p) · `prototype_metrics`(L1/L2 4 列) · `graph_metrics`(重算 11 列) · `extra_metrics`(真实表中新确认的 18 列: corridor_area_ratio, Topological_*, L3_*, s_mean_* 等) · `match`(|q−c|, q−c, log 面积比, 同 cluster)。**`total_score` 绝不作为特征。**

**Ranker 注册表**：`weighted_rule`, `ridge`, `random_forest`, `extra_trees`, `mlp`, `lgbm_regressor`, `lgbm_lambdarank`（LambdaMART, 主推）。

### ⚠️ 关于 weighted_rule 的诚实说明
`weighted_rule` 直接读取候选的 `total_score`（数据库属性），而评估相关度就是 `total_score`，因此它是 **oracle 上界**而非公平基线（合成数据上 NDCG≈1.0 即此原因）。它在系统中的作用是"质量感知排序"的参考线；论文中比较学习型 ranker 时应报告：
1) `weighted_rule(w_similarity=0)` = 纯质量排序上界；2) 学习型 ranker（不见 label）；3) 随机排序下界。
学习型 ranker 的真实价值体现在：**给定新项目条件、在没有 total_score 的情况下预测哪些原型质量高且匹配**。若真实数据上学习型 ranker 显著低于上界但显著高于随机，则说明条件→质量存在可学习但有限的信号（这与审计中 Spearman(total_area, score)=0.48、其余 <0.2 一致）。

## 阶段二：规则 / 搜索 + 计算几何 + 修复
```
generator: rule_expander(subdivide/branch/chord, 按 layout 先验与目标平均度自适应) | search_expander(16 次采样取大纲指标最优)
→ geometry_decoder: corridor_partition (骨架坐标固定+弹簧布局放新节点→等比例缩放入边界→走廊缓冲→中庭→临街带切店铺/纵深切主力店)
→ repairer: basic (连通修复、越界拉回、小店合并、不可达删除；全部记录在 diagnostics.repairs)
→ evaluator: topology_spec(大纲 5 指标) + geometry(边界内比例/重叠率/无效多边形/可达率/面积分布/约束满足率)
→ 候选按 (全部硬检查通过, 约束满足率, 节点偏差) 排序
```
合成语料 120 条上：rule 通过率 96.7%，search 通过率 100%（ASPL 偏差 16.1% → 2.5%）。**真实语料结果待运行**（`scripts/run_real_data_phase2.sh`）。
`target_edge_recall ≈ 58–59%`：对真实扩展的边只能命中约六成，说明规则方法能满足统计指标但未学到数据中的具体连接模式——这正是 Phase 4 学习型生成器（原型保持图编辑）的改进空间与消融对照。
