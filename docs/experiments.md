# 实验记录

所有结果均**待在真实数据上运行**。下表为合成数据 smoke run（仅验证流程，不构成结论）：

| 配置 | 数据 | val NDCG@5 | test NDCG@10 (n_queries) | 备注 |
|---|---|---|---|---|
| stage1/random_forest | synthetic(108 层) | 见 outputs/experiments/stage1/stage1_random_forest/val_metrics.json | 0.99 (10) | 合成标签由密度/ASPL 构造，故偏高 |
| stage2/rule_baseline | synthetic 原型 SYN00000_1, N_target=51 | — | 3/3 候选通过大纲 5 指标（密度偏差 5–15 %，ASPL 偏差 5–16 %）| 边准确率 100 % 由构造保证 |
