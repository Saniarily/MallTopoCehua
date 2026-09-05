"""Stage 1 — topology-prototype retrieval, ranking and explanation.

Pipeline: hard-constraint filtering → retriever (Top-N recall) → ranker (re-rank) →
calibrator (confidence) → explainer (structured evidence). All components are pluggable
through the registry.
"""

from mall_space_planner.stage1.base import (
    BaseCalibrator,
    BaseExplainer,
    BaseGraphEncoder,
    BaseRanker,
    BaseRetriever,
    BaseStage1Model,
    RankingContext,
)

__all__ = [
    "BaseCalibrator",
    "BaseExplainer",
    "BaseGraphEncoder",
    "BaseRanker",
    "BaseRetriever",
    "BaseStage1Model",
    "RankingContext",
]
