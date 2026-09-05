# 架构与第一版技术方案

```
PlanningCondition ─► Stage1Pipeline: HardConstraintFilter → retriever(knn) → ranker(weighted_rule|random_forest|extra_trees|ridge)
                                     → confidence → TemplateExplainer(+counterfactuals) ─► Top-K Recommendation
选定 TopologyPrototype + SiteBoundary + ConstraintSet ─► Stage2Pipeline: generator(rule_expander) → geometry_decoder(skeleton_embed)
                                     → evaluator(topology_spec: 节点偏差/边准确率/密度偏差/ASPL偏差/推理时间) ─► GeneratedLayout (JSON/GeoJSON/SVG)
```
- 所有组件通过 `registry.register(kind, name)` 注册、`registry.build(kind, spec)` 由 YAML 构造；训练脚本不 import 具体模型。
- 配置支持 `_base_` 继承与 `--override a.b=c`；组件 spec 的 `name` 改变时整体替换（避免参数串味）。
- 评估协议（阶段一）：test 楼层为 query，候选 = 同 bucket、**不同 mall** 的 test 楼层；相关度 = 候选 `total_score` 在候选集内 min-max。

## 下一步模型设计（Phase 2–4）
1. **Phase 2**：LightGBM/XGBoost LambdaMART（pairwise/listwise）、MLP ranker、SHAP 解释、多 seed 汇总脚本 `run_ablation.py`/`compare_experiments.py`。
2. **创新点候选**（均带消融开关）：(a) 原型作为可解释中间态；(b) 手工拓扑指标 + 可学习图表征的**残差融合**（GIN/GraphSAGE 旁路，Late Fusion，baseline = 纯表格）；(c) 双塔条件–原型对比检索 + 质量感知重排；(d) 原型保持的可控图编辑（rule_expander → 学习型 op 策略/条件 VAE），并用大纲 5 指标 + 连通率 + 多样性评估；(e) 反事实解释（已实现雏形）。
3. **Phase 3**：店铺分区（shapely 走廊缓冲 → 剩余区域 Voronoi/条带划分）、修复器（越界/重叠/断连）、面积约束、PNG 导出。
4. **Phase 5**：Streamlit Viewer Hub（数据浏览 / 实验中心 / 策划验证），后端逻辑已在 `pipelines` 中与 UI 解耦。
