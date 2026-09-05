"""Rankers (re-ranking stage). All registered under kind ``ranker``."""

from mall_space_planner.stage1.rankers.rule import QualityOracleRanker, RandomRanker, WeightedRuleRanker
from mall_space_planner.stage1.rankers.sklearn_rankers import (
    ExtraTreesPointwiseRanker,
    LinearPointwiseRanker,
    MLPPointwiseRanker,
    RandomForestPointwiseRanker,
)

try:  # optional dependency
    from mall_space_planner.stage1.rankers.lightgbm_rankers import LGBMLambdaRankRanker, LGBMPointwiseRanker
except ImportError:  # pragma: no cover
    LGBMLambdaRankRanker = LGBMPointwiseRanker = None  # type: ignore[assignment,misc]

try:  # optional dependency (torch)
    from mall_space_planner.stage1.rankers.deep_ranker import DeepResidualRanker
except ImportError:  # pragma: no cover
    DeepResidualRanker = None  # type: ignore[assignment,misc]

__all__ = [
    "DeepResidualRanker",
    "WeightedRuleRanker",
    "RandomRanker",
    "QualityOracleRanker",
    "ExtraTreesPointwiseRanker",
    "LinearPointwiseRanker",
    "MLPPointwiseRanker",
    "RandomForestPointwiseRanker",
    "LGBMLambdaRankRanker",
    "LGBMPointwiseRanker",
]
