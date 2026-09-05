# 数据 Schema（Pydantic v2，`src/mall_space_planner/schemas/core.py`）

| 契约 | 关键字段 | 来源/备注 |
|---|---|---|
| `PlanningCondition` | `city_cluster`, `people, GDP_2023, PCDI_2023, TP_2023, mall_area_count, nearest_distance_km, count_1km, count_2km, total_area, Tx`, `preferred_layout` | 字段名 = 旧 `config.yaml` query_cols；单位待确认 |
| `SiteBoundary` | `exterior[(x,y)]`, `holes`, `entrances`, `atrium_hints`; `area()`, `rectangle()` | 新增（旧数据无轮廓）|
| `TopologyGraph` | `adjacency{node:[nbrs]}`, `node_types`, `positions`, `node_attrs`; 自动对称化、去自环 | 旧图 CSV / ShareGPT 邻接表 |
| `TopologyMetrics` | `num_nodes, num_edges, density, avg_degree, diameter, avg_shortest_path, num_cycles, n_components, clustering, degree_entropy, max_betweenness` + 旧 `L1_density, L2_diameter, L2_complexity, L2_integration` | 前者由 `topology.metrics` 重算 |
| `TopologyPrototype` | `prototype_id(=floor_id)`, `graph`, `layout_type`, `metrics`, `quality_score` | |
| `MallCase` | `floor_id, mall_id, floor_index, condition, prototype, total_score, boundary?` | 一行主表 |
| `RankingLabel` | `query_id, candidate_id, relevance, group_id` | bucket 内 min-max 相关度 |
| `Recommendation` / `RecommendationExplanation` | `rank, prototype_id, score, confidence, …`; `recommendation_summary, top_factors, matched_case_evidence, topology_reasoning, risks, counterfactuals, confidence` | 需求书要求的解释 JSON 结构 |
| `ConstraintSet` | `target_num_nodes, target_num_shops, shop_area_min/max/mean, corridor_width, min_entrances, num_atria, target_metrics, layout_type, weights{}` | 阶段二输入 |
| `SpaceUnit` / `GeneratedLayout` | `kind∈{shop,corridor,junction,atrium,entrance}, polygon, area`; `topology, skeleton_positions, units, diagnostics` | 阶段二输出 |
| `EvaluationResult` | `evaluator, metrics, passed, overall_pass, details` | 两阶段共用 |
| `LayoutType` | 一字型/简单环型/多环型/简单集中型/复杂集中型/简单型/Unknown_Layout | ShareGPT 语料原文 |

处理后数据（`CaseDatabase.save`）：`cases.parquet|csv`（含 `split`, `has_graph`, `g_*` 图指标列）+ `graphs.jsonl` + `manifest.json`（列名、划分协议、环境快照）。
