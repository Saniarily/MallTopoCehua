"""Prototype-fidelity evaluation protocol (label-free alternative to the quality protocol).

Motivation (real-data findings, see docs/experiments.md): ``total_score`` is a *mall-level*
rating with a hard ceiling, weakly related to floor topology, and dominated by
``total_area``. Ranking by it cannot show that *topology prototype* recommendation is
useful. This protocol instead asks the question the system is actually built for:

    Given only the planning conditions of a held-out mall, do the recommended prototypes
    resemble the topologies that mall actually built?

For each test mall (query = its conditions, candidates = train-split floors after the same
hard filter used at inference):

* ``type_hit@K``        – any Top-K prototype has the same layout type (``type8``) as any
  floor of the held-out mall;
* ``type_precision@K``  – fraction of Top-K whose layout type occurs in the held-out mall;
* ``metric_dist@K``     – mean standardised Euclidean distance between each Top-K prototype's
  topology descriptors and the *nearest* real floor of the mall (lower = better);
* ``quality@K``         – mean ``total_score`` of Top-K (quality is still reported, not optimised).

Reference lines: ``random`` (uniform over the filtered pool), ``majority`` (most frequent
layout type in the pool), and an ``oracle`` that knows the mall's true types.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mall_space_planner.schemas import PlanningCondition
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _std_matrix(df: pd.DataFrame, cols: list[str], ref: pd.DataFrame) -> np.ndarray:
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mu = ref[cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(dtype=float)
    sd = ref[cols].apply(pd.to_numeric, errors="coerce").std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    x = np.where(np.isnan(x), mu, x)
    return (x - mu) / sd


def evaluate_prototype_fidelity(
    pipeline: Stage1Pipeline,
    split: str = "test",
    ks: tuple[int, ...] = (5, 10),
    metric_cols: list[str] | None = None,
    layout_col: str = "layout_type",
    max_malls: int | None = None,
    seed: int = 0,
    reference: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the protocol. ``reference`` ∈ {None, "random", "majority", "oracle"} replaces the ranker."""
    db = pipeline.ctx.db
    spec = pipeline.features.spec
    test = db.split(split)
    test = test[test["has_graph"]] if "has_graph" in test else test
    pool_all = pipeline.candidate_pool()
    metric_cols = metric_cols or [c for c in (db.metric_cols + db.graph_metric_cols) if c in pool_all.columns]
    rng = np.random.RandomState(seed)
    malls = test[db.mall_id_col].drop_duplicates().tolist()
    rng.shuffle(malls)
    if max_malls:
        malls = malls[:max_malls]
    kmax = max(ks)
    rows: list[dict[str, Any]] = []
    for mall in tqdm(malls, desc=f"fidelity[{split}]", leave=False):
        floors = test[test[db.mall_id_col] == mall]
        true_types = set(floors[layout_col].dropna().astype(str)) if layout_col in floors else set()
        q = floors.iloc[0]
        query = PlanningCondition(**{c: (None if pd.isna(q.get(c)) else q[c]) for c in spec.query_cols + [spec.city_cluster_col] if c in q})
        filt = pipeline.hard_filter.apply(query, pool_all)
        pool = filt.candidates
        if len(pool) < kmax:
            continue
        if reference == "random":
            ranked = pool.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        elif reference == "majority":
            maj = pool[layout_col].mode().iloc[0] if layout_col in pool and pool[layout_col].notna().any() else None
            ranked = pd.concat([pool[pool[layout_col] == maj], pool[pool[layout_col] != maj]]).reset_index(drop=True)
        elif reference == "oracle":
            hit = pool[layout_col].astype(str).isin(true_types) if layout_col in pool else pd.Series(False, index=pool.index)
            ranked = pd.concat([pool[hit], pool[~hit]]).reset_index(drop=True)
        else:
            ranked, _ = pipeline.score_candidates(query, pool_all)
        if len(ranked) < kmax:
            continue
        # topology distance: each recommended prototype vs nearest real floor of the mall
        if metric_cols and len(floors):
            f_m = _std_matrix(floors, metric_cols, pool_all)
            r_m = _std_matrix(ranked.head(kmax), metric_cols, pool_all)
            d = np.linalg.norm(r_m[:, None, :] - f_m[None, :, :], axis=2).min(axis=1)  # [kmax]
        else:
            d = np.full(kmax, np.nan)
        rec_types = ranked[layout_col].astype(str).tolist() if layout_col in ranked else []
        row: dict[str, Any] = {"mall_id": mall, "n_floors": len(floors), "n_pool": len(pool), "true_types": "|".join(sorted(true_types))}
        for k in ks:
            hits = [t in true_types for t in rec_types[:k]]
            row[f"type_hit@{k}"] = float(any(hits)) if true_types else np.nan
            row[f"type_precision@{k}"] = float(np.mean(hits)) if true_types and hits else np.nan
            row[f"metric_dist@{k}"] = float(np.nanmean(d[:k]))
            row[f"quality@{k}"] = float(pd.to_numeric(ranked.head(k)[db.label_col], errors="coerce").mean())
        rows.append(row)
    per_mall = pd.DataFrame(rows)
    if per_mall.empty:
        return per_mall, {"n_malls": 0}
    agg: dict[str, Any] = {"n_malls": int(len(per_mall)), "split": split, "reference": reference or "model"}
    for c in per_mall.columns:
        if "@" in c:
            agg[c] = float(per_mall[c].mean())
            agg[f"{c}_std"] = float(per_mall[c].std(ddof=0))
    return per_mall, agg


def layout_predictability(db, layout_col: str = "layout_type", seed: int = 42) -> dict[str, Any]:  # noqa: ANN001
    """How much layout-type information do planning conditions carry? (LightGBM vs majority).

    Trains a multiclass classifier condition→type8 on train malls (one row per floor) and
    reports accuracy / macro-F1 on test floors, against the majority-class baseline. A
    result close to majority means conditions alone cannot pick a layout type — which
    bounds what any Stage-1 recommender can achieve on ``type_hit``.
    """
    from sklearn.metrics import accuracy_score, f1_score

    cols = db.query_cols + ["city_cluster"]
    tr, te = db.split("train"), db.split("test")
    tr = tr[tr[layout_col].notna()]
    te = te[te[layout_col].notna()]
    classes = sorted(tr[layout_col].astype(str).unique())
    cid = {c: i for i, c in enumerate(classes)}
    ytr = tr[layout_col].astype(str).map(cid).to_numpy()
    yte = te[layout_col].astype(str).map(cid).fillna(-1).astype(int).to_numpy()
    xtr = tr[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xte = te[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    maj = int(np.bincount(ytr).argmax())
    out: dict[str, Any] = {"classes": classes, "n_train": int(len(tr)), "n_test": int(len(te)), "majority_class": classes[maj], "majority_accuracy": float((yte == maj).mean())}
    try:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)
        clf.fit(xtr, ytr)
        pred = clf.predict(xte)
        out["lgbm_accuracy"] = float(accuracy_score(yte, pred))
        out["lgbm_macro_f1"] = float(f1_score(yte, pred, average="macro"))
        out["lgbm_feature_importance"] = dict(zip(cols, [float(v) for v in clf.feature_importances_], strict=True))
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(np.nan_to_num(xtr), ytr)
        pred = clf.predict(np.nan_to_num(xte))
        out["rf_accuracy"] = float(accuracy_score(yte, pred))
        out["rf_macro_f1"] = float(f1_score(yte, pred, average="macro"))
    # Floor-level features (area/Tx) added: does the *site* tell more than the *city*?
    return out
