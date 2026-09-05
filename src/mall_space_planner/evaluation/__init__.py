"""Evaluation: ranking metrics (Stage 1) and topology/geometry evaluators (Stage 2)."""

from mall_space_planner.evaluation.ranking_metrics import (
    average_precision,
    hit_rate_at_k,
    kendall_tau,
    mrr,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    recall_at_k,
    spearman,
)
from mall_space_planner.evaluation.stage1_eval import evaluate_stage1
from mall_space_planner.evaluation.geometry_eval import GeometryEvaluator
from mall_space_planner.evaluation.stage2_eval import TopologySpecEvaluator

__all__ = [
    "average_precision",
    "hit_rate_at_k",
    "kendall_tau",
    "mrr",
    "ndcg_at_k",
    "pairwise_accuracy",
    "precision_at_k",
    "recall_at_k",
    "spearman",
    "evaluate_stage1",
    "TopologySpecEvaluator",
    "GeometryEvaluator",
]
