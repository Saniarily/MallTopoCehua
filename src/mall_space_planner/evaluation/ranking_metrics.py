"""Ranking / retrieval metrics on a single query (aggregate with ``np.mean`` outside).

Conventions: ``scores`` are model outputs (higher = better), ``relevance`` are graded
labels in ``[0, 1]`` (continuous) and ``relevant`` is a boolean mask for binary metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="stable")


def dcg_at_k(rels_in_pred_order: np.ndarray, k: int) -> float:
    r = np.asarray(rels_in_pred_order, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return float(np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=float)
    pred = rel[_order(scores)]
    ideal = np.sort(rel)[::-1]
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(pred, k) / idcg if idcg > 0 else 0.0


def precision_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    rel = np.asarray(relevant, dtype=bool)[_order(scores)][:k]
    return float(rel.mean()) if rel.size else 0.0


def recall_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    rel_all = np.asarray(relevant, dtype=bool)
    n_rel = rel_all.sum()
    if n_rel == 0:
        return 0.0
    return float(rel_all[_order(scores)][:k].sum() / n_rel)


def hit_rate_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    rel = np.asarray(relevant, dtype=bool)[_order(scores)][:k]
    return float(rel.any())


def average_precision(scores: np.ndarray, relevant: np.ndarray) -> float:
    rel = np.asarray(relevant, dtype=bool)[_order(scores)]
    if rel.sum() == 0:
        return 0.0
    hits = np.cumsum(rel)
    prec = hits / np.arange(1, rel.size + 1)
    return float((prec * rel).sum() / rel.sum())


def mrr(scores: np.ndarray, relevant: np.ndarray) -> float:
    rel = np.asarray(relevant, dtype=bool)[_order(scores)]
    idx = np.flatnonzero(rel)
    return float(1.0 / (idx[0] + 1)) if idx.size else 0.0


def spearman(scores: np.ndarray, relevance: np.ndarray) -> float:
    if len(scores) < 3 or np.std(scores) < 1e-12 or np.std(relevance) < 1e-12:
        return 0.0
    r = stats.spearmanr(scores, relevance).correlation
    return float(0.0 if np.isnan(r) else r)


def kendall_tau(scores: np.ndarray, relevance: np.ndarray) -> float:
    if len(scores) < 3 or np.std(scores) < 1e-12 or np.std(relevance) < 1e-12:
        return 0.0
    r = stats.kendalltau(scores, relevance).correlation
    return float(0.0 if np.isnan(r) else r)


def pairwise_accuracy(scores: np.ndarray, relevance: np.ndarray, max_pairs: int = 5000, seed: int = 0) -> float:
    """Fraction of (i, j) pairs with rel_i > rel_j whose scores are ordered correctly."""
    s, r = np.asarray(scores, dtype=float), np.asarray(relevance, dtype=float)
    n = len(s)
    if n < 2:
        return 0.0
    ii, jj = np.triu_indices(n, k=1)
    mask = r[ii] != r[jj]
    ii, jj = ii[mask], jj[mask]
    if ii.size == 0:
        return 0.0
    if ii.size > max_pairs:
        rng = np.random.RandomState(seed)
        sel = rng.choice(ii.size, max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    correct = np.sign(s[ii] - s[jj]) == np.sign(r[ii] - r[jj])
    return float(correct.mean())
