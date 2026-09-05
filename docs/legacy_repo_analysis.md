# 旧仓库分析：MallTopoRanker（只读审计，2026-09-05）

> 图例：**[事实]** = 从代码/数据文件直接确认；**[推断]** = 基于证据的判断；**[假设]** = 待实验验证。
> 自动生成的原始审计数据见 `outputs/reports/legacy_repo_audit.{md,json}`（`python scripts/audit_legacy_repo.py --legacy-repo <path>`）。

## 1. 目录结构 [事实]
```
MallTopoRanker/
├── config.yaml                 # 数据路径、10个query列、4个metric列、bucket阈值、模型/训练超参
├── requirements.txt            # torch>=2.1, torch-geometric>=2.5, pandas, numpy, sklearn, pyyaml, tqdm, matplotlib
├── README_run.md               # 仅 4 行（不完整）
├── sweep_*.yaml (10 个)        # 消融/超参扫描配置
├── src/ (18 个 .py, 3413 行)   # train_ranker(_v0/_inbatch).py, model_ranker.py, model_components.py,
│                               # dataset_pairs.py, infer_ranker(_v0).py, evaluate_ranker.py, explain.py,
│                               # run_ablation.py, plot_paper_figures.py, utils_*.py, query_cases.json
├── cache/graphs_pt/            # 4977 个 PyG Data 缓存（floor_id.pt），1157 个 mall
├── cache/scaler.pkl            # 已拟合的 StandardScaler（q 10维 / m 4维，n=3983 训练样本）
└── outputs/                    # checkpoints(399 .pt)、train_logs、16 次 ablation 结果、figures
```
Git 历史仅 1 个 commit（`init: MallTopoRanker baseline`）。

## 2. 数据入口 [事实]
- 主表：`DATASET_3+6类城市_with_rkn_ratios_and_topo_metrics.csv`（本沙箱中**不存在**，仅在用户 Mac 上）。
- 图目录：`Graph_Data/total_graph_data/total_graph_data(6005个地上层)`，每层两个文件：
  `{floor_id}_M_simplified.csv`（列 `Source,Target`）与 `{floor_id}_M_simplified_node_attributes.csv`
  （列 `Node_ID, Total_L_Neighbors[, CenterPoint="(x, y)"]`）。
- `floor_id = {mall_id}_{floor_index}`（如 `B000A08791_3`），节点 id 全部为 `M###`（走廊交点）。
- 主表列（由 config + 代码确认）：`floor_id, mall_id, city_cluster, total_score`，query 列
  `people, GDP_2023, PCDI_2023, TP_2023, mall_area_count, nearest_distance_km, count_1km, count_2km, total_area, Tx`，
  metric 列 `L1_density, L2_diameter, L2_complexity, L2_integration`。**其他列无法在沙箱内确认。**

## 3. 缓存图统计 [事实]（4977 图）
| 项 | 值 |
|---|---|
| 节点数 min / p25 / 中位 / p75 / max | 2 / 11 / 19 / 30 / 94（均值 21.6）|
| 边数 中位 / 均值 / max | 22 / 24.8 / 108 |
| 每 mall 楼层数 | 1–7 层，主要 3–5 层 |
| 节点特征 | `[Total_L_Neighbors, x_norm, y_norm]`（坐标按图中心化并归一到 [-1,1]）|

## 4. 特征尺度（来自 scaler.pkl，训练集 3983 样本）[事实]
people 9602±9738；GDP_2023 57695±36489；PCDI_2023 56311±21664；TP_2023 5005±3412；mall_area_count 101±74；
nearest_distance_km 0.215±0.329；count_1km 11.0±8.6；count_2km 27.8±19.1；total_area 122125±102814；Tx 27.7±16.7；
L1_density 0.091±0.058；L2_diameter 25.9±12.3；L2_complexity 1.064±0.052；L2_integration 0.467±0.109。
**单位/语义（尤其 `Tx`、`TP_2023`）文档中未定义，需数据方确认。**

## 5. 旧模型结构 [事实]
`GraphMatchRanker`：10 个 query 特征 → 逐特征 token（id_emb + value_proj）→ 2 层 TransformerEncoder；
候选图 → 3 层 GraphSAGE（节点特征 3 维）+ 4 个 metric token；query tokens 对 [metric tokens + node tokens] 做
多头 cross-attention；拼接 `[q, g, cross, |q−g|, q*g]` → MLP 打分。损失：pairwise logistic loss，
pair 采样在 bucket（city_cluster × 面积档 <200k / 200k–450k / ≥450k）内进行，query 取 bucket 均值或正样本自身 query。

## 6. 旧实验结果 [事实]
- 主训练日志（20 epoch）：val NDCG@10 **第 1 个 epoch 最高 0.678**，之后随 train loss 下降而持续恶化（0.54–0.61），
  val pairwise acc 停在 0.63–0.67 → 典型过拟合，模型几乎没学到可泛化信号。
- 16 次消融（d_model/gnn_layers/dropout/wd/scheduler/query 策略/bucket 策略）：best NDCG@10 中位约 0.67–0.70，
  seed 44 上可达 0.80+，**不同 seed 之间差异（0.66 vs 0.80）远大于任何因素之间的差异**。
- Stage C 对比 `use_bucket_query` True/False：0.668±0.005 vs 0.675±0.009，无显著差异。

## 7. 为何 Transformer+GNN+Cross-Attention 效果不佳 [推断]
1. **数据规模与模型复杂度不匹配**：~4000 训练楼层、每 bucket 内候选同质，却用 Transformer + 3 层 GNN + cross-attn；val 曲线第 1 epoch 即最优。
2. **监督信号弱且与 query 无关**：训练 pair 的 query 是 bucket 平均或正样本自身的 query，而候选来自同 bucket，
   因此 query 特征在 bucket 内几乎无区分力（同 mall 楼层共享全部 city 级特征），模型本质只在学 `total_score` 的排序，
   query 塔基本是噪声输入 → 这解释了 use_bucket_query 无差异。
3. **图输入信息量低**：节点特征只有 `Total_L_Neighbors` + 归一化坐标；4 个 L1/L2 指标已高度概括图结构，GNN 边际收益小。
4. **评估协议波动大**：val NDCG 每 bucket 仅抽 20 个 query / 300 候选，候选相关度用 bucket 内 min-max，导致 seed 方差极大。
5. **cross-attention 解释被当作因果**：`explain.py` 直接把 attention 权重当"条件→指标驱动关系"，不可靠。
6. **未与传统基线比较**：仓库内没有任何 kNN / 树模型 / 线性基线，无法判断深度模型是否真的带来增益。

## 8. 数据划分与泄漏 [事实 + 推断]
- 按 `mall_id` 分组划分（正确，避免同 mall 楼层跨集）。
- 但 `train_ranker.py` 用 `split_seed`(默认 42) 划分，`evaluate_ranker.py` 用 `train.seed` 划分：**当 sweep 修改 seed 时训练/评估划分不一致 → 可能泄漏**。
- 评估候选集为同 bucket 的其他 test 样本，未排除同 mall → 同 mall 其他楼层作为候选带来"近重复"泄漏（新仓库已排除）。
- `pairs_per_bucket` 上限与 `len(idxs)*10` 交互，使小 bucket 欠采样。

## 9. 对新仓库的结论
- 复用：数据格式（CSV/图文件）、bucket 业务规则、按 mall 分组划分协议、bucket 内 min-max 相关度定义。
- 不复用：模型代码、attention 解释、PyG 缓存格式。
- 优先建立 kNN / 规则加权 / 树模型 / 线性 / LTR 基线，再在同一评估协议下检验 GNN/双塔是否有增益。
