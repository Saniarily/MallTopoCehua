"""Type-conditional quality model: *given* planning conditions, which layout type is expected
to be rated higher?

Scientific framing
------------------
Layout type is a *design decision variable*, not a function of the site/city conditions.
The question the planner asks is therefore the conditional-effect question

    E[score | conditions, layout_type]   for each candidate type,

which encodes the empirical finding that the best-rated layout type differs across city
clusters (type × cluster interaction). This module fits that model with tree ensembles
(which capture interactions natively), scores every layout type for a new project, and
returns a ranked list with bootstrap uncertainty and the empirical evidence behind it.

Evaluation (``evaluate_type_recommender``)
* predictive: RMSE / Spearman of predicted vs actual score on held-out malls;
* incremental value of the type variable: Δ over a conditions-only model (ablation);
* per-cluster type ordering: Kendall τ between predicted mean score per type and the
  empirical mean score per type in the held-out set;
* policy value: mean actual score of held-out floors whose real type equals the
  recommended type vs. the others (same cluster), i.e. an off-policy uplift estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.features.builders import HEAVY_TAIL_DEFAULT, _ColumnScaler
from mall_space_planner.registry import register
from mall_space_planner.schemas import LAYOUT_TYPES, LayoutType, PlanningCondition
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TypeRecommendation:
    layout_type: str
    expected_score: float
    ci_low: float
    ci_high: float
    rank: int
    n_comparable_cases: int
    empirical_mean_score: float | None
    share_in_comparable: float | None


@dataclass
class TypeRecommenderResult:
    recommendations: list[TypeRecommendation]
    conditions_only_score: float
    cluster: int | None
    evidence: dict[str, Any] = field(default_factory=dict)


class BaseTypeRecommender:
    def fit(self, db: CaseDatabase, train_df: pd.DataFrame) -> BaseTypeRecommender:  # noqa: D401
        raise NotImplementedError

    def recommend(self, db: CaseDatabase, query: PlanningCondition) -> TypeRecommenderResult:
        raise NotImplementedError


@register("stage1_model", "type_conditional_trees")
class TreeTypeRecommender(BaseTypeRecommender):
    """ExtraTrees on [standardised conditions, one-hot type]; bootstrap ensemble for CIs."""

    def __init__(self, n_estimators: int = 400, min_samples_leaf: int = 5, n_bootstrap: int = 20, seed: int = 42, types: list[str] | None = None, use_topology_metrics: bool = False) -> None:
        self.n_estimators, self.min_samples_leaf, self.n_bootstrap, self.seed = n_estimators, min_samples_leaf, n_bootstrap, seed
        self.types = [t for t in (types or LAYOUT_TYPES) if t != LayoutType.UNKNOWN.value]
        self.use_topology_metrics = use_topology_metrics
        self.models_: list[ExtraTreesRegressor] = []
        self.cond_only_: ExtraTreesRegressor | None = None
        self.scaler_: _ColumnScaler | None = None
        self.cond_cols_: list[str] = []
        self.cluster_col_ = "city_cluster"
        self.empirical_: pd.DataFrame | None = None  # per (cluster, type): n, mean score

    # ------------------------------------------------------------------ features
    def _x(self, df: pd.DataFrame, types: pd.Series | str) -> np.ndarray:
        cond = self.scaler_.transform(df)
        clus = pd.to_numeric(df[self.cluster_col_], errors="coerce").fillna(-1).to_numpy()[:, None]
        if isinstance(types, str):
            oh = np.tile((np.array(self.types) == types).astype(float), (len(df), 1))
        else:
            oh = np.stack([(types.astype(str).to_numpy() == t).astype(float) for t in self.types], axis=1)
        return np.concatenate([cond, clus, oh], axis=1)

    def _xc(self, df: pd.DataFrame) -> np.ndarray:
        return np.concatenate([self.scaler_.transform(df), pd.to_numeric(df[self.cluster_col_], errors="coerce").fillna(-1).to_numpy()[:, None]], axis=1)

    # ------------------------------------------------------------------ fit
    def fit(self, db: CaseDatabase, train_df: pd.DataFrame) -> TreeTypeRecommender:
        self.cond_cols_ = list(db.query_cols)
        self.cluster_col_ = db.manifest.get("city_cluster_col", "city_cluster")
        df = train_df[train_df[db.label_col].notna() & train_df["layout_type"].notna()]
        df = df[df["layout_type"].astype(str).isin(self.types)]
        self.scaler_ = _ColumnScaler(self.cond_cols_, list(HEAVY_TAIL_DEFAULT)).fit(df)
        x, y = self._x(df, df["layout_type"]), df[db.label_col].astype(float).to_numpy()
        rng = np.random.RandomState(self.seed)
        self.models_ = []
        for b in range(self.n_bootstrap):
            idx = rng.choice(len(x), len(x), replace=True)
            m = ExtraTreesRegressor(n_estimators=max(20, self.n_estimators // self.n_bootstrap), min_samples_leaf=self.min_samples_leaf, n_jobs=-1, random_state=self.seed + b)
            self.models_.append(m.fit(x[idx], y[idx]))
        self.cond_only_ = ExtraTreesRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, n_jobs=-1, random_state=self.seed).fit(self._xc(df), y)
        self.empirical_ = df.groupby([self.cluster_col_, "layout_type"])[db.label_col].agg(n="size", mean="mean").reset_index()
        logger.info("TreeTypeRecommender fitted on %d floors, %d types, %d bootstrap models", len(df), len(self.types), self.n_bootstrap)
        return self

    # ------------------------------------------------------------------ predict
    def predict_all(self, df: pd.DataFrame) -> np.ndarray:
        """[n, n_types, n_bootstrap] predicted scores for every row and every type."""
        out = np.zeros((len(df), len(self.types), len(self.models_)))
        for j, t in enumerate(self.types):
            x = self._x(df, t)
            for b, m in enumerate(self.models_):
                out[:, j, b] = m.predict(x)
        return out

    def predict_actual(self, df: pd.DataFrame) -> np.ndarray:
        x = self._x(df, df["layout_type"])
        return np.mean([m.predict(x) for m in self.models_], axis=0)

    def recommend(self, db: CaseDatabase, query: PlanningCondition) -> TypeRecommenderResult:
        row = pd.DataFrame([{**{c: getattr(query, c, None) for c in self.cond_cols_}, self.cluster_col_: query.city_cluster}])
        pred = self.predict_all(row)[0]  # [types, boot]
        mean, lo, hi = pred.mean(1), np.percentile(pred, 10, axis=1), np.percentile(pred, 90, axis=1)
        cond_only = float(self.cond_only_.predict(self._xc(row))[0])
        emp = self.empirical_[self.empirical_[self.cluster_col_] == query.city_cluster] if self.empirical_ is not None and query.city_cluster is not None else None
        total = int(emp["n"].sum()) if emp is not None and len(emp) else 0
        order = np.argsort(-mean)
        recs = []
        for r, j in enumerate(order):
            t = self.types[j]
            e = emp[emp["layout_type"] == t] if emp is not None else None
            n = int(e["n"].iloc[0]) if e is not None and len(e) else 0
            recs.append(TypeRecommendation(t, float(mean[j]), float(lo[j]), float(hi[j]), r + 1, n, float(e["mean"].iloc[0]) if n else None, (n / total) if total else None))
        return TypeRecommenderResult(recs, cond_only, query.city_cluster, {"model": "type_conditional_trees", "n_bootstrap": len(self.models_)})


# --------------------------------------------------------------------------- evaluation
def evaluate_type_recommender(rec: TreeTypeRecommender, db: CaseDatabase, split: str = "test") -> dict[str, Any]:
    te = db.split(split)
    te = te[te[db.label_col].notna() & te["layout_type"].astype(str).isin(rec.types)].reset_index(drop=True)
    y = te[db.label_col].astype(float).to_numpy()
    pred_actual = rec.predict_actual(te)
    pred_cond = rec.cond_only_.predict(rec._xc(te))
    out: dict[str, Any] = {"n_test_floors": int(len(te)), "n_test_malls": int(te[db.mall_id_col].nunique())}
    out["rmse_with_type"] = float(np.sqrt(np.mean((pred_actual - y) ** 2)))
    out["rmse_conditions_only"] = float(np.sqrt(np.mean((pred_cond - y) ** 2)))
    out["spearman_with_type"] = float(stats.spearmanr(pred_actual, y).correlation)
    out["spearman_conditions_only"] = float(stats.spearmanr(pred_cond, y).correlation)
    # per-cluster type ordering agreement + policy value
    all_pred = rec.predict_all(te).mean(2)  # [n, types]
    best_idx = all_pred.argmax(1)
    te = te.assign(_rec_type=[rec.types[i] for i in best_idx], _pred_actual=pred_actual)
    per_cluster = []
    for c, g in te.groupby(rec.cluster_col_):
        emp = g.groupby("layout_type")[db.label_col].mean()
        pm = pd.Series(all_pred[g.index].mean(0), index=rec.types)
        common = [t for t in emp.index if t in pm.index]
        tau = float(stats.kendalltau(emp[common], pm[common]).correlation) if len(common) >= 3 else np.nan
        match = g["layout_type"].astype(str) == g["_rec_type"]
        per_cluster.append({"cluster": c, "n": int(len(g)), "kendall_tau_type_order": tau, "empirical_best_type": str(emp.idxmax()), "recommended_best_type": str(pm[common].idxmax()) if common else None, "score_when_type_matches_rec": float(g.loc[match, db.label_col].mean()) if match.any() else np.nan, "score_when_type_differs": float(g.loc[~match, db.label_col].mean()) if (~match).any() else np.nan, "n_match": int(match.sum())})
    out["per_cluster"] = per_cluster
    taus = [p["kendall_tau_type_order"] for p in per_cluster if not np.isnan(p["kendall_tau_type_order"])]
    out["mean_kendall_tau_type_order"] = float(np.mean(taus)) if taus else np.nan
    m = te["layout_type"].astype(str) == te["_rec_type"]
    out["policy_uplift"] = float(te.loc[m, db.label_col].mean() - te.loc[~m, db.label_col].mean()) if m.any() and (~m).any() else np.nan
    out["best_type_agreement_rate"] = float(np.mean([p["empirical_best_type"] == p["recommended_best_type"] for p in per_cluster])) if per_cluster else np.nan
    return out
