"""Two-level Stage-1 pipeline: hard filter → retriever (recall) → ranker (re-rank) → explain.

The pipeline is configured entirely from YAML (``configs/stage1/*.yaml``)::

    stage1:
      hard_filter: {area_thresholds: [200000, 450000], min_candidates: 5}
      retriever: {name: knn, params: {metric: euclidean}}
      recall_top_n: 200
      ranker: {name: random_forest, params: {n_estimators: 200}}
      explainer: {name: template}
      counterfactuals: {enabled: true, deltas: {total_area: 0.3, count_1km: 0.5}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.features.builders import FeatureSpec, TabularFeatureBuilder
from mall_space_planner.registry import build
from mall_space_planner.schemas import (
    LayoutType,
    PlanningCondition,
    Recommendation,
    TopologyMetrics,
)
from mall_space_planner.stage1.base import BaseExplainer, BaseRanker, BaseRetriever, RankingContext
from mall_space_planner.stage1.retrievers.hard_filter import HardConstraintFilter
from mall_space_planner.utils.logging import get_logger
from mall_space_planner.utils.repro import collect_environment_info

logger = get_logger(__name__)


class Stage1Pipeline:
    """Fit on a :class:`CaseDatabase`, then answer ``recommend`` queries with explanations."""

    def __init__(self, config: dict[str, Any], db: CaseDatabase) -> None:
        self.config = config
        s1 = config.get("stage1", {})
        feat_cfg = dict(config.get("features", {}))
        spec = FeatureSpec(
            query_cols=db.query_cols,
            metric_cols=db.metric_cols,
            graph_metric_cols=db.graph_metric_cols,
            **{k: v for k, v in feat_cfg.items() if k in FeatureSpec.__dataclass_fields__ and k not in ("query_cols", "metric_cols", "graph_metric_cols")},
        )
        self.features = TabularFeatureBuilder(spec)
        self.ctx = RankingContext(db=db, features=self.features, config=config)

        self.hard_filter = HardConstraintFilter(**(s1.get("hard_filter") or {}))
        self.retriever: BaseRetriever | None = build("retriever", s1["retriever"]) if s1.get("retriever") else None
        self.recall_top_n = int(s1.get("recall_top_n", 200))
        self.ranker: BaseRanker = build("ranker", s1.get("ranker", {"name": "weighted_rule"}))
        self.explainer: BaseExplainer = build("explainer", s1.get("explainer", {"name": "template"}))
        self.cf_cfg = s1.get("counterfactuals", {"enabled": True})
        self.candidate_splits = list(s1.get("candidate_splits", ["train"]))
        self.fitted = False
        self.fit_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ fit
    def fit(self) -> Stage1Pipeline:
        db = self.ctx.db
        train_df = db.split("train")
        train_df = train_df[train_df["has_graph"]].reset_index(drop=True) if "has_graph" in train_df else train_df
        val_df = db.split("val") if "split" in db.cases else None
        t0 = time.perf_counter()
        self.features.fit(train_df)
        if self.retriever is not None:
            self.retriever.fit(self.ctx, train_df)
        self.ranker.fit(self.ctx, train_df, val_df)
        self.fitted = True
        self.fit_info = {
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)) if val_df is not None else 0,
            "fit_seconds": time.perf_counter() - t0,
            "ranker": getattr(self.ranker, "registry_name", type(self.ranker).__name__),
            "retriever": getattr(self.retriever, "registry_name", None),
            "history": self.ranker.training_history(),
        }
        logger.info("Stage1Pipeline fitted: %s", {k: v for k, v in self.fit_info.items() if k != "history"})
        return self

    # ------------------------------------------------------------------ candidates
    def candidate_pool(self, exclude_mall_ids: set[str] | None = None) -> pd.DataFrame:
        df = self.ctx.candidates
        if "split" in df and self.candidate_splits:
            df = df[df["split"].isin(self.candidate_splits)]
        if exclude_mall_ids:
            df = df[~df[self.ctx.db.mall_id_col].isin(exclude_mall_ids)]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ scoring
    def _query_df(self, query: PlanningCondition, n: int) -> pd.DataFrame:
        cols = self.features.spec.query_cols + [self.features.spec.city_cluster_col]
        rec = {c: getattr(query, c, None) for c in cols}
        return pd.DataFrame([rec] * n)

    def score_candidates(self, query: PlanningCondition, pool: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return the ranked frame (with ``score``) and the evidence dict for the explainer."""
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted")
        filt = self.hard_filter.apply(query, pool)
        cand = filt.candidates
        if self.retriever is not None and len(cand) > 0:
            cand = self.retriever.retrieve(self.ctx, query, cand, top_n=self.recall_top_n)
        if len(cand) == 0:
            return cand.assign(score=[]), {"constraints_applied": filt.applied, "constraints_relaxed": filt.relaxed, "note": "empty candidate set"}
        q_df = self._query_df(query, len(cand))
        scores = self.ranker.score(self.ctx, q_df, cand)
        ranked = cand.assign(score=scores).sort_values("score", ascending=False).reset_index(drop=True)
        evidence = {
            "constraints_applied": filt.applied,
            "constraints_relaxed": filt.relaxed,
            "feature_importance": self.ranker.feature_importance(),
            "n_after_filter": int(len(filt.candidates)),
            "n_after_recall": int(len(cand)),
        }
        return ranked, evidence

    def _confidence(self, ranked: pd.DataFrame, i: int) -> float:
        """Heuristic confidence: score margin over the pool, squashed to [0, 1].

        A learned calibrator (Phase 4) can replace this via the ``calibrator`` component.
        """
        s = ranked["score"].to_numpy(dtype=float)
        if len(s) < 2 or np.std(s) < 1e-9:
            return 0.5
        z = (s[i] - s.mean()) / (s.std() + 1e-9)
        return float(1.0 / (1.0 + np.exp(-z)))

    def _counterfactuals(self, query: PlanningCondition, pool: pd.DataFrame, target_id: str, base_rank: int) -> list[dict[str, Any]]:
        if not self.cf_cfg or not self.cf_cfg.get("enabled", True):
            return []
        deltas: dict[str, float] = self.cf_cfg.get("deltas") or {"total_area": 0.3, "count_1km": 0.5, "PCDI_2023": 0.2, "nearest_distance_km": 0.5}
        out: list[dict[str, Any]] = []
        for feat, rel in deltas.items():
            base = getattr(query, feat, None)
            if base is None:
                continue
            for direction in (+1, -1):
                new_val = float(base) * (1.0 + direction * rel)
                q2 = query.model_copy(update={feat: new_val})
                try:
                    ranked2, _ = self.score_candidates(q2, pool)
                except Exception:  # noqa: BLE001
                    continue
                ids = ranked2[self.ctx.db.id_col].astype(str).tolist()
                new_rank = ids.index(target_id) + 1 if target_id in ids else None
                out.append({"feature": feat, "change": f"{'+' if direction > 0 else '-'}{int(rel * 100)}%", "new_value": new_val, "rank_before": base_rank, "rank_after": new_rank})
        return out

    # ------------------------------------------------------------------ public API
    def recommend(
        self,
        query: PlanningCondition,
        top_k: int = 10,
        exclude_mall_ids: set[str] | None = None,
        with_explanations: bool = True,
        with_counterfactuals: bool = True,
    ) -> list[Recommendation]:
        t0 = time.perf_counter()
        pool = self.candidate_pool(exclude_mall_ids)
        ranked, evidence = self.score_candidates(query, pool)
        db = self.ctx.db
        recs: list[Recommendation] = []
        for i in range(min(top_k, len(ranked))):
            row = ranked.iloc[i]
            fid = str(row[db.id_col])
            conf = self._confidence(ranked, i)
            ev = dict(evidence, confidence=conf)
            if with_explanations and with_counterfactuals and i < 3:
                ev["counterfactuals"] = self._counterfactuals(query, pool, fid, i + 1)
            metrics = TopologyMetrics(**{c: (None if pd.isna(row.get(c)) else float(row[c])) for c in db.metric_cols if c in row})
            for c in db.graph_metric_cols:
                if c in row and not pd.isna(row[c]):
                    setattr(metrics, c.replace("g_", "", 1), float(row[c]))
            lt = row.get("layout_type")
            recs.append(
                Recommendation(
                    rank=i + 1,
                    prototype_id=fid,
                    score=float(row["score"]),
                    confidence=conf,
                    quality_score=None if pd.isna(row.get(db.label_col)) else float(row[db.label_col]),
                    similarity=None if "similarity" not in row or pd.isna(row["similarity"]) else float(row["similarity"]),
                    layout_type=LayoutType(lt) if isinstance(lt, str) and lt in {t.value for t in LayoutType} else None,
                    metrics=metrics,
                    explanation=self.explainer.explain(self.ctx, query, ranked, i, ev) if with_explanations else None,
                )
            )
        logger.debug("recommend: %d results in %.3fs", len(recs), time.perf_counter() - t0)
        return recs

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path) -> Path:
        import joblib

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"ranker": self.ranker, "retriever": self.retriever, "features": self.features.state_dict()}, out_dir / "stage1_model.joblib")
        with open(out_dir / "checkpoint_meta.json", "w", encoding="utf-8") as f:
            json.dump({"config": self.config, "fit_info": self.fit_info, "environment": collect_environment_info()}, f, ensure_ascii=False, indent=2, default=str)
        return out_dir

    @classmethod
    def load(cls, ckpt_dir: str | Path, db: CaseDatabase, config: dict[str, Any] | None = None) -> Stage1Pipeline:
        import joblib

        ckpt_dir = Path(ckpt_dir)
        with open(ckpt_dir / "checkpoint_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = config or meta["config"]
        obj = cls(cfg, db)
        blob = joblib.load(ckpt_dir / "stage1_model.joblib")
        obj.ranker = blob["ranker"]
        obj.retriever = blob["retriever"]
        obj.features = TabularFeatureBuilder.from_state_dict(blob["features"])
        obj.ctx = RankingContext(db=db, features=obj.features, config=cfg)
        obj.fit_info = meta.get("fit_info", {})
        obj.fitted = True
        return obj
