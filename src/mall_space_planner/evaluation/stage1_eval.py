"""Stage-1 evaluation protocol (leakage-free).

For every *test* floor used as a query, the candidate list is the set of **other test
floors from a different mall** inside the same bucket (city_cluster × area-bin), exactly
like the legacy protocol but with an explicit same-mall exclusion. Graded relevance is
the candidate's ``total_score`` min-max normalised within the candidate list; binary
relevance marks the top ``binary_top_frac`` fraction.

Returns per-query rows and an aggregate dictionary (mean ± std across queries) so that
multi-seed experiments can be pooled downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mall_space_planner.evaluation.ranking_metrics import (
    average_precision,
    hit_rate_at_k,
    kendall_tau,
    mrr,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    recall_at_k,
    spearman,
)
from mall_space_planner.schemas import PlanningCondition
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.stage1.rankers.sklearn_rankers import make_bucket_id
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def evaluate_stage1(
    pipeline: Stage1Pipeline,
    split: str = "test",
    ks: tuple[int, ...] = (5, 10, 20),
    binary_top_frac: float = 0.2,
    max_queries: int | None = None,
    min_candidates: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    db = pipeline.ctx.db
    spec = pipeline.features.spec
    df = db.split(split)
    if "has_graph" in df:
        df = df[df["has_graph"]].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"split {split!r} is empty")
    thresholds = pipeline.hard_filter.area_thresholds
    df = df.assign(_bucket=make_bucket_id(df, spec.city_cluster_col, spec.total_area_col, thresholds))
    rng = np.random.RandomState(seed)
    query_rows = df.sample(frac=1.0, random_state=rng).head(max_queries) if max_queries else df

    rows: list[dict[str, Any]] = []
    for _, q in tqdm(query_rows.iterrows(), total=len(query_rows), desc=f"eval[{split}]", leave=False):
        cand = df[(df["_bucket"] == q["_bucket"]) & (df[db.mall_id_col] != q[db.mall_id_col])]
        if len(cand) < min_candidates:
            continue
        query = PlanningCondition(**{c: (None if pd.isna(q.get(c)) else q[c]) for c in spec.query_cols + [spec.city_cluster_col] if c in q})
        # Score directly (bypassing hard filter so that the candidate set is fixed by protocol).
        q_df = pipeline._query_df(query, len(cand))
        cand_r = cand.reset_index(drop=True)
        if pipeline.retriever is not None:
            cand_r = pipeline.retriever.retrieve(pipeline.ctx, query, cand_r, top_n=len(cand_r))
            q_df = pipeline._query_df(query, len(cand_r))
        scores = pipeline.ranker.score(pipeline.ctx, q_df, cand_r)
        y = cand_r[db.label_col].astype(float).to_numpy()
        rel = _minmax(y)
        n_pos = max(1, int(np.ceil(binary_top_frac * len(rel))))
        relevant = np.zeros(len(rel), dtype=bool)
        relevant[np.argsort(-rel)[:n_pos]] = True
        r: dict[str, Any] = {"query_id": q[db.id_col], "bucket": q["_bucket"], "n_candidates": len(cand_r)}
        for k in ks:
            r[f"ndcg@{k}"] = ndcg_at_k(scores, rel, k)
            r[f"precision@{k}"] = precision_at_k(scores, relevant, k)
            r[f"recall@{k}"] = recall_at_k(scores, relevant, k)
            r[f"hit@{k}"] = hit_rate_at_k(scores, relevant, k)
        r["map"] = average_precision(scores, relevant)
        r["mrr"] = mrr(scores, relevant)
        r["spearman"] = spearman(scores, rel)
        r["kendall_tau"] = kendall_tau(scores, rel)
        r["pairwise_acc"] = pairwise_accuracy(scores, rel)
        rows.append(r)

    per_query = pd.DataFrame(rows)
    if per_query.empty:
        logger.warning("No evaluable queries (buckets too small). Lower min_candidates or check the split.")
        return per_query, {"n_queries": 0}
    metric_cols = [c for c in per_query.columns if c not in ("query_id", "bucket", "n_candidates")]
    agg: dict[str, Any] = {"n_queries": int(len(per_query)), "split": split}
    for c in metric_cols:
        agg[c] = float(per_query[c].mean())
        agg[f"{c}_std"] = float(per_query[c].std(ddof=0))
    return per_query, agg
