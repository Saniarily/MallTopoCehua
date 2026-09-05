"""kNN / similarity retriever over standardised planning conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mall_space_planner.registry import register
from mall_space_planner.schemas import PlanningCondition
from mall_space_planner.stage1.base import BaseRetriever, RankingContext


@register("retriever", "knn")
class KNNRetriever(BaseRetriever):
    """Cosine / Euclidean similarity between the query condition and candidate conditions.

    Optional per-feature weights let the rule-based configuration emphasise, e.g.,
    ``total_area`` and competition features.
    """

    def __init__(self, metric: str = "euclidean", feature_weights: dict[str, float] | None = None) -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.metric = metric
        self.feature_weights = feature_weights or {}
        self._w: np.ndarray | None = None

    def fit(self, ctx: RankingContext, train_df: pd.DataFrame) -> KNNRetriever:
        cols = ctx.features.spec.query_cols
        self._w = np.array([float(self.feature_weights.get(c, 1.0)) for c in cols], dtype=np.float32)
        return self

    def _query_df(self, query: PlanningCondition, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame([{c: getattr(query, c, None) for c in cols}])

    def retrieve(self, ctx: RankingContext, query: PlanningCondition, candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
        cols = ctx.features.spec.query_cols
        w = self._w if self._w is not None else np.ones(len(cols), dtype=np.float32)
        q = ctx.features.condition_matrix(self._query_df(query, cols)) * w
        c = ctx.features.condition_matrix(candidates) * w
        if self.metric == "cosine":
            qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
            sim = (cn @ qn.T).ravel()
        else:
            d = np.linalg.norm(c - q, axis=1)
            sim = 1.0 / (1.0 + d)
        out = candidates.copy()
        out["similarity"] = sim
        out = out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
        return out
