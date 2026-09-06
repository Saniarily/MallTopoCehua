| experiment | variant | ranker | seeds | ndcg@5 | ndcg@10 | map | spearman | pairwise_acc |
|---|---|---|---|---|---|---|---|---|
| stage1_lgbm_lambdarank | no_graph_metrics | lgbm_lambdarank | 3 | 0.695 ± 0.022 | 0.675 ± 0.011 | 0.392 ± 0.010 | 0.292 ± 0.023 | 0.659 ± 0.009 |
| stage1_lgbm_lambdarank | no_extra_metrics | lgbm_lambdarank | 3 | 0.682 ± 0.017 | 0.675 ± 0.020 | 0.366 ± 0.030 | 0.232 ± 0.036 | 0.621 ± 0.023 |
| stage1_lgbm_lambdarank | no_legacy_metrics | lgbm_lambdarank | 3 | 0.681 ± 0.039 | 0.682 ± 0.033 | 0.371 ± 0.036 | 0.306 ± 0.029 | 0.669 ± 0.018 |
| stage1_lgbm_lambdarank | no_condition | lgbm_lambdarank | 3 | 0.673 ± 0.016 | 0.665 ± 0.006 | 0.379 ± 0.013 | 0.276 ± 0.008 | 0.643 ± 0.008 |
| stage1_lgbm_lambdarank | no_match | lgbm_lambdarank | 3 | 0.673 ± 0.012 | 0.662 ± 0.012 | 0.343 ± 0.009 | 0.221 ± 0.017 | 0.612 ± 0.003 |
| stage1_lgbm_lambdarank | full | lgbm_lambdarank | 3 | 0.667 ± 0.029 | 0.662 ± 0.019 | 0.377 ± 0.021 | 0.283 ± 0.028 | 0.651 ± 0.016 |
| stage1_lgbm_lambdarank | no_retriever | lgbm_lambdarank | 3 | 0.667 ± 0.029 | 0.662 ± 0.019 | 0.365 ± 0.017 | 0.283 ± 0.028 | 0.651 ± 0.017 |