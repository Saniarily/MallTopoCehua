| experiment | variant | ranker | seeds | ndcg@5 | ndcg@10 | map | spearman | pairwise_acc |
|---|---|---|---|---|---|---|---|---|
| ref_quality_oracle_upper_bound | compare | quality_oracle | 3 | 0.751 ± 0.000 | 0.751 ± 0.000 | 0.808 ± 0.000 | 0.751 ± 0.000 | 0.963 ± 0.000 |
| stage1_rule_knn | compare | weighted_rule | 3 | 0.726 ± 0.000 | 0.727 ± 0.000 | 0.642 ± 0.000 | 0.642 ± 0.000 | 0.927 ± 0.000 |
| stage1_extra_trees | compare | extra_trees | 3 | 0.702 ± 0.012 | 0.689 ± 0.012 | 0.448 ± 0.011 | 0.292 ± 0.006 | 0.673 ± 0.005 |
| stage1_ridge | compare | ridge | 3 | 0.699 ± 0.011 | 0.674 ± 0.004 | 0.463 ± 0.002 | 0.333 ± 0.001 | 0.693 ± 0.001 |
| stage1_lgbm_regressor | compare | lgbm_regressor | 3 | 0.688 ± 0.021 | 0.677 ± 0.016 | 0.452 ± 0.021 | 0.320 ± 0.010 | 0.680 ± 0.005 |
| stage1_deep_residual | compare | deep_residual | 3 | 0.681 ± 0.032 | 0.672 ± 0.019 | 0.430 ± 0.014 | 0.304 ± 0.012 | 0.677 ± 0.007 |
| stage1_lgbm_lambdarank_valuegrade | compare | lgbm_lambdarank | 3 | 0.667 ± 0.029 | 0.662 ± 0.019 | 0.377 ± 0.021 | 0.283 ± 0.028 | 0.651 ± 0.016 |
| stage1_lgbm_lambdarank | compare | lgbm_lambdarank | 3 | 0.656 ± 0.006 | 0.643 ± 0.012 | 0.426 ± 0.008 | 0.339 ± 0.006 | 0.679 ± 0.005 |
| stage1_random_forest | compare | random_forest | 3 | 0.626 ± 0.054 | 0.627 ± 0.040 | 0.403 ± 0.011 | 0.228 ± 0.009 | 0.627 ± 0.004 |
| stage1_mlp | compare | mlp | 3 | 0.558 ± 0.088 | 0.575 ± 0.047 | 0.326 ± 0.024 | 0.190 ± 0.021 | 0.609 ± 0.015 |
| ref_random_lower_bound | compare | random | 3 | 0.521 ± 0.004 | 0.522 ± 0.003 | 0.244 ± 0.005 | -0.002 ± 0.000 | 0.481 ± 0.001 |