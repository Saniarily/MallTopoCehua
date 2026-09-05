"""Pydantic data contracts shared by every module.

The field names for legacy data were confirmed by reading the old ``MallTopoRanker``
repository (``config.yaml``, ``src/*.py``, cached ``scaler.pkl`` and ``graphs_pt``) and
the Stage-2 ShareGPT corpus. See ``docs/data_schema.md`` for provenance per field.
"""

from mall_space_planner.schemas.core import (
    LAYOUT_TYPES,
    ConstraintSet,
    EvaluationResult,
    GeneratedLayout,
    LayoutType,
    MallCase,
    PlanningCondition,
    RankingLabel,
    Recommendation,
    RecommendationExplanation,
    SiteBoundary,
    SpaceUnit,
    TopologyGraph,
    TopologyMetrics,
    TopologyPrototype,
)

__all__ = [
    "LAYOUT_TYPES",
    "ConstraintSet",
    "EvaluationResult",
    "GeneratedLayout",
    "LayoutType",
    "MallCase",
    "PlanningCondition",
    "RankingLabel",
    "Recommendation",
    "RecommendationExplanation",
    "SiteBoundary",
    "SpaceUnit",
    "TopologyGraph",
    "TopologyMetrics",
    "TopologyPrototype",
]
