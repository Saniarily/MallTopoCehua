"""LightGBM rankers: LambdaMART (listwise Learning-to-Rank) and pointwise regression.

Why LambdaMART here
-------------------
The real label ``total_score`` is a *mall-level* rating (identical for all floors of a
mall, 21 unique values in [2.9, 4.9], strongly left-skewed). Pointwise regression on such
a saturated label wastes capacity on the ceiling; LambdaMART only cares about the
*order* of candidates within a query group, which is exactly the recommendation objective.

Groups = training query floors; candidates = other floors of the same bucket from
**different malls**; graded relevance = within-group min-max of ``total_score`` mapped to
integer grades ``0..n_grades-1`` (LightGBM lambdarank needs integer labels).

Per-candidate SHAP-style contributions are obtained natively via ``pred_contrib=True``,
so no extra dependency is needed for the explainer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.registry import register
from mall_space_planner.stage1.base import BaseRanker, RankingContext
from mall_space_planner.stage1.rankers.sklearn_rankers import make_bucket_id, sample_groups
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _lgb():  # noqa: ANN202
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lightgbm is required for this ranker: pip install lightgbm") from exc
    return lgb


class _LGBMBase(BaseRanker):
    supports_feature_importance = True

    def __init__(
        self,
        candidates_per_query: int = 30,
        n_grades: int = 5,
        seed: int = 42,
        area_thresholds: list[float] | None = None,
        early_stopping_rounds: int = 50,
        **model_params: Any,
    ) -> None:
        self.candidates_per_query = candidates_per_query
        self.n_grades = n_grades
        self.seed = seed
        self.area_thresholds = list(area_thresholds or [200_000, 450_000])
        self.early_stopping_rounds = early_stopping_rounds
        self.model_params = model_params
        self.model = None
        self.feature_names_: list[str] = []
        self.history_: dict[str, list[float]] = {}
        self.best_iteration_: int | None = None

    # ------------------------------------------------------------------ data
    def _build(self, ctx: RankingContext, df: pd.DataFrame, rng: np.random.RandomState, n_cand: int):
        spec = ctx.features.spec
        bucket = make_bucket_id(df, spec.city_cluster_col, spec.total_area_col, self.area_thresholds)
        q_df, c_df, rel, groups = sample_groups(df, ctx.db.label_col, ctx.db.mall_id_col, bucket, n_cand, rng)
        if len(rel) == 0:
            return None
        x, names = ctx.features.pair_features(q_df, c_df)
        grades = np.clip(np.round(rel * (self.n_grades - 1)), 0, self.n_grades - 1).astype(int)
        return x, rel, grades, groups, names

    def _make_params(self, objective: str) -> dict[str, Any]:
        p = {
            "objective": objective,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": self.seed,
            "num_threads": 0,
        }
        if objective == "lambdarank":
            p.update({"metric": "ndcg", "eval_at": [5, 10], "label_gain": [float(2**g - 1) for g in range(self.n_grades)]})
        else:
            p.update({"metric": "l2"})
        p.update(self.model_params)
        return p

    # ------------------------------------------------------------------ fit
    def _fit_impl(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None, objective: str, num_rounds: int) -> None:
        lgb = _lgb()
        rng = np.random.RandomState(self.seed)
        built = self._build(ctx, train_df, rng, self.candidates_per_query)
        if built is None:
            raise ValueError("No training groups could be built (buckets too small)")
        x, rel, grades, groups, names = built
        self.feature_names_ = names
        label = grades if objective == "lambdarank" else rel
        dtrain = lgb.Dataset(x, label=label, group=groups if objective == "lambdarank" else None, feature_name=[n.replace(":", "_") for n in names])
        valid_sets, valid_names = [dtrain], ["train"]
        if val_df is not None and len(val_df) > 3:
            vb = self._build(ctx, val_df, rng, min(20, self.candidates_per_query))
            if vb is not None:
                vx, vrel, vgr, vgroups, _ = vb
                dval = lgb.Dataset(vx, label=vgr if objective == "lambdarank" else vrel, group=vgroups if objective == "lambdarank" else None, reference=dtrain)
                valid_sets.append(dval)
                valid_names.append("val")
        evals: dict = {}
        callbacks = [lgb.record_evaluation(evals)]
        if len(valid_sets) > 1 and self.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.model = lgb.train(self._make_params(objective), dtrain, num_boost_round=num_rounds, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
        self.best_iteration_ = self.model.best_iteration or None
        self.history_ = {f"{s}_{m}": [float(v) for v in vals] for s, d in evals.items() for m, vals in d.items()}
        logger.info("%s fitted: %d rows, %d groups, %d features, best_iter=%s", type(self).__name__, len(x), len(groups), x.shape[1], self.best_iteration_)

    def score(self, ctx: RankingContext, query_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
        x, _ = ctx.features.pair_features(query_df, cand_df)
        return self.model.predict(x, num_iteration=self.best_iteration_).astype(np.float32)

    def contributions(self, ctx: RankingContext, query_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
        """Per-row SHAP-style feature contributions ``[n, n_features + 1]`` (last column = bias)."""
        x, _ = ctx.features.pair_features(query_df, cand_df)
        return np.asarray(self.model.predict(x, num_iteration=self.best_iteration_, pred_contrib=True))

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None:
            return None
        imp = self.model.feature_importance(importance_type="gain")
        total = float(imp.sum()) or 1.0
        return {n: float(v) / total for n, v in zip(self.feature_names_, imp, strict=False)}

    def training_history(self) -> dict[str, list[float]]:
        return self.history_


@register("ranker", "lgbm_lambdarank")
class LGBMLambdaRankRanker(_LGBMBase):
    """Listwise Learning-to-Rank (LambdaMART)."""

    def __init__(self, num_rounds: int = 500, **kw: Any) -> None:
        super().__init__(**kw)
        self.num_rounds = num_rounds

    def fit(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> LGBMLambdaRankRanker:
        self._fit_impl(ctx, train_df, val_df, "lambdarank", self.num_rounds)
        return self


@register("ranker", "lgbm_regressor")
class LGBMPointwiseRanker(_LGBMBase):
    """Pointwise gradient boosting on graded relevance (ablation partner of LambdaMART)."""

    def __init__(self, num_rounds: int = 500, **kw: Any) -> None:
        super().__init__(**kw)
        self.num_rounds = num_rounds

    def fit(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> LGBMPointwiseRanker:
        self._fit_impl(ctx, train_df, val_df, "regression", self.num_rounds)
        return self
