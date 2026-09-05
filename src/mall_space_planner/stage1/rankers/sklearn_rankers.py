"""Pointwise tabular rankers built on scikit-learn (strong, dependency-light baselines).

Training protocol
-----------------
For each training *query* floor we sample candidates from the same bucket
(``city_cluster`` × area-bin) and build pair features with
:class:`TabularFeatureBuilder`. The regression target is the candidate's graded
relevance: its ``total_score`` min-max normalised **within the bucket**. This is the
same supervision the legacy pairwise model used, expressed pointwise so that any
regressor can serve as ranker. Learning-to-rank (LambdaMART via LightGBM/XGBoost) is
added in Phase 2 with the identical feature pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from mall_space_planner.registry import register
from mall_space_planner.stage1.base import BaseRanker, RankingContext
from mall_space_planner.stage1.retrievers.hard_filter import area_bin
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


def make_bucket_id(df: pd.DataFrame, city_col: str, area_col: str, thresholds: list[float]) -> pd.Series:
    bins = df[area_col].apply(lambda x: area_bin(x, thresholds))
    return df[city_col].astype(str) + "_" + bins.astype(str)


def sample_pairs(
    df: pd.DataFrame,
    label_col: str,
    bucket: pd.Series,
    pairs_per_query: int,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return aligned (query rows, candidate rows, graded relevance) frames."""
    q_idx: list[int] = []
    c_idx: list[int] = []
    rel: list[float] = []
    for _, g in df.groupby(bucket):
        if len(g) < 3:
            continue
        y = g[label_col].astype(float).to_numpy()
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        scale = (hi - lo) if hi > lo else 1.0
        idx = g.index.to_numpy()
        for qi in idx:
            others = idx[idx != qi]
            take = rng.choice(others, size=min(pairs_per_query, len(others)), replace=False)
            for ci in take:
                q_idx.append(qi)
                c_idx.append(ci)
                rel.append((float(df.at[ci, label_col]) - lo) / scale)
    return df.loc[q_idx].reset_index(drop=True), df.loc[c_idx].reset_index(drop=True), np.asarray(rel, dtype=np.float32)


class _SklearnPointwiseRanker(BaseRanker):
    supports_feature_importance = True

    def __init__(self, pairs_per_query: int = 20, seed: int = 42, area_thresholds: list[float] | None = None, **model_params: Any) -> None:
        self.pairs_per_query = pairs_per_query
        self.seed = seed
        self.area_thresholds = list(area_thresholds or [200_000, 450_000])
        self.model_params = model_params
        self.model = self._make_model(**model_params)
        self.feature_names_: list[str] = []
        self.history_: dict[str, list[float]] = {}

    def _make_model(self, **params: Any):  # noqa: ANN202
        raise NotImplementedError

    def fit(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> _SklearnPointwiseRanker:
        rng = np.random.RandomState(self.seed)
        spec = ctx.features.spec
        bucket = make_bucket_id(train_df, spec.city_cluster_col, spec.total_area_col, self.area_thresholds)
        q_df, c_df, rel = sample_pairs(train_df, ctx.db.label_col, bucket, self.pairs_per_query, rng)
        if len(rel) == 0:
            raise ValueError("No training pairs could be sampled (buckets too small)")
        x, names = ctx.features.pair_features(q_df, c_df)
        self.feature_names_ = names
        self.model.fit(x, rel)
        train_mse = float(np.mean((self.model.predict(x) - rel) ** 2))
        self.history_ = {"train_mse": [train_mse]}
        if val_df is not None and len(val_df) > 3:
            vb = make_bucket_id(val_df, spec.city_cluster_col, spec.total_area_col, self.area_thresholds)
            vq, vc, vrel = sample_pairs(val_df, ctx.db.label_col, vb, min(10, self.pairs_per_query), rng)
            if len(vrel):
                vx, _ = ctx.features.pair_features(vq, vc)
                self.history_["val_mse"] = [float(np.mean((self.model.predict(vx) - vrel) ** 2))]
        logger.info("%s fitted on %d pairs (%d features); history=%s", type(self).__name__, len(rel), x.shape[1], self.history_)
        return self

    def score(self, ctx: RankingContext, query_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
        x, _ = ctx.features.pair_features(query_df, cand_df)
        return self.model.predict(x).astype(np.float32)

    def feature_importance(self) -> dict[str, float] | None:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            coef = getattr(self.model, "coef_", None)
            if coef is None:
                return None
            imp = np.abs(coef)
        return {n: float(v) for n, v in zip(self.feature_names_, imp, strict=False)}

    def training_history(self) -> dict[str, list[float]]:
        return self.history_


@register("ranker", "random_forest")
class RandomForestPointwiseRanker(_SklearnPointwiseRanker):
    def _make_model(self, **params: Any) -> RandomForestRegressor:
        defaults = {"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": -1, "random_state": self.seed}
        defaults.update(params)
        return RandomForestRegressor(**defaults)


@register("ranker", "extra_trees")
class ExtraTreesPointwiseRanker(_SklearnPointwiseRanker):
    def _make_model(self, **params: Any) -> ExtraTreesRegressor:
        defaults = {"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": -1, "random_state": self.seed}
        defaults.update(params)
        return ExtraTreesRegressor(**defaults)


@register("ranker", "ridge")
class LinearPointwiseRanker(_SklearnPointwiseRanker):
    def _make_model(self, **params: Any) -> Ridge:
        defaults = {"alpha": 1.0}
        defaults.update(params)
        return Ridge(**defaults)
