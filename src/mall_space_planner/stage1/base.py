"""Abstract interfaces for Stage-1 components.

All concrete implementations are registered under kinds ``retriever``, ``ranker``,
``graph_encoder``, ``calibrator``, ``explainer`` and ``stage1_model``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.features.builders import TabularFeatureBuilder
from mall_space_planner.schemas import PlanningCondition, RecommendationExplanation


@dataclass
class RankingContext:
    """Everything a component may need at fit/predict time."""

    db: CaseDatabase
    features: TabularFeatureBuilder
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> pd.DataFrame:
        """Candidate pool = cases that have a graph (any split; filtering happens in pipeline)."""
        df = self.db.cases
        return df[df["has_graph"]] if "has_graph" in df else df


class _Persistable(ABC):
    """save/load contract shared by all components (joblib-based by default)."""

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path):  # noqa: ANN206
        import joblib

        return joblib.load(path)


class BaseRetriever(_Persistable):
    """Recall stage: return Top-N candidate ids with similarity scores."""

    @abstractmethod
    def fit(self, ctx: RankingContext, train_df: pd.DataFrame) -> BaseRetriever: ...

    @abstractmethod
    def retrieve(self, ctx: RankingContext, query: PlanningCondition, candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Return ``candidates`` subset with an added ``similarity`` column, sorted descending."""


class BaseRanker(_Persistable):
    """Re-ranking stage: score (query, candidate) pairs."""

    supports_feature_importance: bool = False

    @abstractmethod
    def fit(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> BaseRanker: ...

    @abstractmethod
    def score(self, ctx: RankingContext, query_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
        """Return one score per aligned (query_df[i], cand_df[i]) row."""

    def feature_importance(self) -> dict[str, float] | None:
        return None

    def training_history(self) -> dict[str, list[float]] | None:
        return None


class BaseGraphEncoder(_Persistable):
    """Optional learned graph embedding for prototypes (Phase 4)."""

    @abstractmethod
    def fit(self, ctx: RankingContext, train_df: pd.DataFrame) -> BaseGraphEncoder: ...

    @abstractmethod
    def encode(self, ctx: RankingContext, floor_ids: list[str]) -> np.ndarray: ...


class BaseCalibrator(_Persistable):
    """Map raw scores to [0, 1] confidence."""

    @abstractmethod
    def fit(self, scores: np.ndarray, targets: np.ndarray) -> BaseCalibrator: ...

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray: ...


class BaseExplainer(ABC):
    """Produce structured evidence for a recommendation."""

    @abstractmethod
    def explain(
        self,
        ctx: RankingContext,
        query: PlanningCondition,
        ranked: pd.DataFrame,
        rank_index: int,
        model_evidence: dict[str, Any] | None = None,
    ) -> RecommendationExplanation: ...


class BaseStage1Model(ABC):
    """End-to-end Stage-1 model (retriever + ranker + calibrator + explainer)."""

    @abstractmethod
    def fit(self, ctx: RankingContext) -> BaseStage1Model: ...

    @abstractmethod
    def recommend(self, ctx: RankingContext, query: PlanningCondition, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]: ...
