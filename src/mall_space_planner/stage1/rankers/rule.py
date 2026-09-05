"""Rule-based weighted ranker (transparent baseline).

score = w_quality · quality_norm + w_similarity · similarity  (both in [0, 1])

``quality_norm`` is the candidate's legacy ``total_score`` min-max normalised on the
training split; ``similarity`` is the retriever similarity when present, otherwise a
Gaussian kernel on the standardised condition distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mall_space_planner.registry import register
from mall_space_planner.stage1.base import BaseRanker, RankingContext


@register("ranker", "weighted_rule")
class WeightedRuleRanker(BaseRanker):
    supports_feature_importance = True

    def __init__(self, w_quality: float = 0.6, w_similarity: float = 0.4, sim_bandwidth: float = 2.0) -> None:
        self.w_quality = w_quality
        self.w_similarity = w_similarity
        self.sim_bandwidth = sim_bandwidth
        self.y_min_: float = 0.0
        self.y_max_: float = 1.0

    def fit(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> WeightedRuleRanker:
        y = train_df[ctx.db.label_col].astype(float)
        self.y_min_, self.y_max_ = float(y.min()), float(y.max())
        if self.y_max_ - self.y_min_ < 1e-9:
            self.y_max_ = self.y_min_ + 1.0
        return self

    def score(self, ctx: RankingContext, query_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
        q_norm = (cand_df[ctx.db.label_col].astype(float).fillna(self.y_min_).to_numpy() - self.y_min_) / (self.y_max_ - self.y_min_)
        if "similarity" in cand_df:
            sim = cand_df["similarity"].astype(float).to_numpy()
        else:
            q = ctx.features.condition_matrix(query_df)
            c = ctx.features.condition_matrix(cand_df)
            d2 = ((q - c) ** 2).sum(axis=1)
            sim = np.exp(-d2 / (2 * self.sim_bandwidth**2))
        return (self.w_quality * np.clip(q_norm, 0, 1) + self.w_similarity * np.clip(sim, 0, 1)).astype(np.float32)

    def feature_importance(self) -> dict[str, float]:
        return {"quality_score": self.w_quality, "condition_similarity": self.w_similarity}
