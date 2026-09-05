# 数据审计报告（Phase 0）

> 状态：真实主表 CSV 和图目录**不在本沙箱**（仅在用户 Mac：见 `configs/data/legacy.yaml`）。以下"事实"来自旧仓库缓存
> （4977 图 + scaler）、上传的 `sharegpt_data.json`（20.5 MB）及测试大纲。在用户机器上运行
> `python scripts/prepare_data.py --config configs/data/legacy.yaml` 会在 `data/processed/legacy/audit.md` 生成完整的真实数据审计
> （缺失率、重复、标签分布、within/between-mall 方差、泄漏检查、图连通性）。

## A. 阶段一数据（案例库）
| 项 | 结论 | 类型 |
|---|---|---|
| 样本数 | 主表 ≈ 6005 地上楼层（大纲称 1210 商场 / 6203 楼层含地下）；旧缓存 4977 层有图，1157 商场；scaler 训练集 3983 | 事实 |
| 图节点/边定义 | 节点 = 走廊交点 M；边 = 走廊连接；无向 | 事实 |
| 拓扑原型定义 | 每个楼层的 `_M_simplified` 图即一个原型；未做聚类 | 事实（设计决策）|
| 质量评分 `total_score` | 存在；分布在沙箱不可见。旧代码把它当连续分级相关度 | 事实/待补 |
| 策划条件字段 | 10 列（见 legacy_repo_analysis §4）；**均为城市/商场级，楼层间恒定** → 同 mall 楼层的 query 完全相同 | 事实 + 推断 |
| 场地轮廓 | **主表和图文件中不存在多边形轮廓**；节点仅有像素级 `CenterPoint` | 事实 |
| 图像 | 无 | 事实 |
| 重复 | 待真实数据验证（脚本已实现：feature-row duplicate 检查）| 待运行 |
| 泄漏风险 | 旧评估未排除同 mall 候选；train/eval 划分 seed 可能不一致 | 事实 |
| 可用监督 | pointwise：total_score；pairwise/listwise：bucket 内 min-max 相关度 | 事实 |
| 不足以支持 | 楼层级策划条件；场地轮廓条件下的几何监督；店铺面积监督 | 推断 |

## B. 阶段二数据（sharegpt_data.json）[事实，全量 5632 条解析]
- 5632 条 skeleton → complete_topology 对；节点为字母标签（A–Z, AA…）；邻接表对称。
- Layout 分布：一字型 2101、多环型 1370、简单环型 1046、复杂集中型 397、简单集中型 319、简单型 2、**Unknown_Layout 397**（同时 City=Beijing 英文、Area=0 → 为另一来源的批次，需标注）。
- 27 个城市；Target_Scale 4–305 节点（中位 30）；Area 中位 85200 m²。
- **骨架边在目标中保留率 = 100 %**（min 1.0）→ 边准确率指标对"复制骨架"策略是平凡的，必须配合密度/ASPL 指标。
- 目标节点数 / 骨架节点数 ≈ 1.87（中位 1.53）；|N_gen − N_target|/N_target 均值 13.5 %，最大 80 % → 训练数据本身就有 13 % 的"节点偏差"，模型不应被要求超过数据。
- 仅 79 % 的目标图连通 → 21 % 的 ground truth 不连通，评估时 ASPL 需按最大连通子图计算（已实现）。
- **该语料没有坐标、没有店铺类型（A/D/E/G）、没有面积** → 阶段二几何落位没有监督，只能用规则/优化 + 自定义几何指标。

## C. 建议补充的数据
1. 楼层轮廓多边形（或 total.csv 的空间边界列）→ 场地约束与几何监督；
2. 店铺节点 (A/D/E/G/L) 的面积与邻接（大纲提到 `total.csv`, `M&L.csv`）→ 店铺生成监督；
3. `Tx`、`TP_2023`、`people` 的单位与口径；`total_score` 的评分方法与评分者；
4. 楼层级条件（楼层号、层高、是否地下）以打破"同 mall 楼层 query 相同"的退化。

## D. 已实现的审计工具
`mall_space_planner.data.audit.audit_case_database`：缺失率、重复、标签分布/偏度、within/between-mall 标签方差、
query 特征在 mall 内恒定比例、Spearman(feature, label)、city_cluster/layout 计数、split 计数与三向泄漏检查、图规模与不连通率。
