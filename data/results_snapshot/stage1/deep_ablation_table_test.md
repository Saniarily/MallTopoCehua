| experiment | variant | ranker | seeds | ndcg@5 | ndcg@10 | map | spearman | pairwise_acc |
|---|---|---|---|---|---|---|---|---|
| stage1_deep_residual | no_gnn | deep_residual | 3 | 0.704 ± 0.011 | 0.682 ± 0.005 | 0.439 ± 0.008 | 0.311 ± 0.007 | 0.677 ± 0.008 |
| stage1_deep_residual | mlp_instead_of_transformer | deep_residual | 3 | 0.682 ± 0.003 | 0.676 ± 0.005 | 0.428 ± 0.014 | 0.297 ± 0.009 | 0.671 ± 0.007 |
| stage1_deep_residual | full_residual_tf_gnn | deep_residual | 3 | 0.672 ± 0.045 | 0.668 ± 0.024 | 0.432 ± 0.015 | 0.306 ± 0.010 | 0.678 ± 0.006 |
| stage1_deep_residual | no_smallsample_tricks | deep_residual | 3 | 0.657 ± 0.055 | 0.650 ± 0.028 | 0.428 ± 0.006 | 0.288 ± 0.024 | 0.669 ± 0.015 |
| stage1_deep_residual | no_residual_end2end | deep_residual | 3 | 0.631 ± 0.041 | 0.645 ± 0.024 | 0.419 ± 0.023 | 0.259 ± 0.012 | 0.652 ± 0.007 |