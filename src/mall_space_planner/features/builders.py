"""Tabular feature builders for query (condition), candidate (prototype) and match features.

Three feature blocks are produced (each can be disabled from config for ablations):

* ``condition``   – planning-condition columns of the *query* (standardised, log1p for heavy tails)
* ``prototype``   – hand-crafted topology metrics of the *candidate* (legacy 4 + recomputed graph metrics)
* ``match``       – interaction features: |q − c|, q − c and q / c for condition columns
  shared by query and candidate (candidates carry their own project's conditions), plus
  area ratio and same-cluster indicator.

Scalers are fitted on the training split only and persisted with the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.registry import register

HEAVY_TAIL_DEFAULT: tuple[str, ...] = (
    "people",
    "GDP_2023",
    "PCDI_2023",
    "TP_2023",
    "mall_area_count",
    "count_1km",
    "count_2km",
    "total_area",
    "nearest_distance_km",
)


@dataclass
class FeatureSpec:
    query_cols: list[str]
    metric_cols: list[str]
    graph_metric_cols: list[str] = field(default_factory=list)
    extra_metric_cols: list[str] = field(default_factory=list)
    log1p_cols: list[str] = field(default_factory=lambda: list(HEAVY_TAIL_DEFAULT))
    use_condition: bool = True
    use_prototype_metrics: bool = True
    use_graph_metrics: bool = True
    use_extra_metrics: bool = True
    use_match: bool = True
    city_cluster_col: str = "city_cluster"
    total_area_col: str = "total_area"


class _ColumnScaler:
    """Per-column standardisation with optional log1p and NaN → train-median imputation."""

    def __init__(self, cols: list[str], log1p_cols: list[str]) -> None:
        self.cols = list(cols)
        self.log1p = set(log1p_cols)
        self.median_: dict[str, float] = {}
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def _pre(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.reindex(columns=self.cols)
        x = x.apply(lambda col: pd.to_numeric(col, errors="coerce")).astype(float)
        for c in self.cols:
            if c in self.log1p:
                x[c] = np.log1p(x[c].clip(lower=0))
        return x

    def fit(self, df: pd.DataFrame) -> _ColumnScaler:
        x = self._pre(df)
        for c in self.cols:
            col = x[c]
            self.median_[c] = float(col.median()) if col.notna().any() else 0.0
            filled = col.fillna(self.median_[c])
            self.mean_[c] = float(filled.mean())
            self.std_[c] = float(filled.std(ddof=0)) or 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._pre(df)
        out = np.zeros((len(x), len(self.cols)), dtype=np.float32)
        for j, c in enumerate(self.cols):
            out[:, j] = ((x[c].fillna(self.median_[c]) - self.mean_[c]) / self.std_[c]).to_numpy()
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"cols": self.cols, "log1p": sorted(self.log1p), "median": self.median_, "mean": self.mean_, "std": self.std_}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _ColumnScaler:
        s = cls(d["cols"], d["log1p"])
        s.median_, s.mean_, s.std_ = d["median"], d["mean"], d["std"]
        return s


@register("feature_builder", "tabular")
class TabularFeatureBuilder:
    """Builds fixed-length vectors for (query, candidate) pairs."""

    def __init__(self, spec: FeatureSpec | dict[str, Any]) -> None:
        self.spec = spec if isinstance(spec, FeatureSpec) else FeatureSpec(**spec)
        s = self.spec
        self.cond_scaler = _ColumnScaler(s.query_cols, s.log1p_cols)
        proto_cols = (
            (s.metric_cols if s.use_prototype_metrics else [])
            + (s.graph_metric_cols if s.use_graph_metrics else [])
            + (s.extra_metric_cols if s.use_extra_metrics else [])
        )
        self.proto_scaler = _ColumnScaler(proto_cols, [])
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, train_df: pd.DataFrame) -> TabularFeatureBuilder:
        self.cond_scaler.fit(train_df)
        if self.proto_scaler.cols:
            self.proto_scaler.fit(train_df)
        self.fitted = True
        return self

    # ------------------------------------------------------------------ blocks
    def condition_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return self.cond_scaler.transform(df)

    def prototype_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if not self.proto_scaler.cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return self.proto_scaler.transform(df)

    def match_matrix(self, q: np.ndarray, c: np.ndarray, q_df: pd.DataFrame, c_df: pd.DataFrame) -> np.ndarray:
        """Interaction features between standardised query and candidate conditions."""
        diff = q - c
        blocks = [np.abs(diff), diff]
        s = self.spec
        extra = []
        if s.total_area_col in q_df and s.total_area_col in c_df:
            qa = q_df[s.total_area_col].astype(float).to_numpy()
            ca = c_df[s.total_area_col].astype(float).to_numpy()
            ratio = np.log((qa + 1.0) / (ca + 1.0))
            extra.append(np.nan_to_num(ratio, nan=0.0)[:, None])
        if s.city_cluster_col in q_df and s.city_cluster_col in c_df:
            same = (q_df[s.city_cluster_col].to_numpy() == c_df[s.city_cluster_col].to_numpy()).astype(np.float32)
            extra.append(same[:, None])
        if extra:
            blocks.append(np.concatenate(extra, axis=1))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    # ------------------------------------------------------------------ pairs
    def pair_features(self, q_df: pd.DataFrame, c_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Feature matrix for aligned rows of ``q_df`` (queries) and ``c_df`` (candidates)."""
        assert len(q_df) == len(c_df), "q_df and c_df must be aligned row-wise"
        s = self.spec
        q = self.condition_matrix(q_df)
        c_cond = self.condition_matrix(c_df)
        parts: list[np.ndarray] = []
        names: list[str] = []
        if s.use_condition:
            parts.append(q)
            names += [f"q:{c}" for c in s.query_cols]
        proto = self.prototype_matrix(c_df)
        if proto.shape[1]:
            parts.append(proto)
            names += [f"c:{c}" for c in self.proto_scaler.cols]
        if s.use_match:
            m = self.match_matrix(q, c_cond, q_df, c_df)
            parts.append(m)
            names += [f"absdiff:{c}" for c in s.query_cols] + [f"diff:{c}" for c in s.query_cols]
            extra_n = m.shape[1] - 2 * len(s.query_cols)
            names += ["log_area_ratio", "same_cluster"][:extra_n]
        if not parts:
            raise ValueError("All feature blocks disabled")
        return np.concatenate(parts, axis=1), names

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.__dict__, "cond": self.cond_scaler.to_dict(), "proto": self.proto_scaler.to_dict(), "fitted": self.fitted}

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> TabularFeatureBuilder:
        fb = cls(FeatureSpec(**d["spec"]))
        fb.cond_scaler = _ColumnScaler.from_dict(d["cond"])
        fb.proto_scaler = _ColumnScaler.from_dict(d["proto"])
        fb.fitted = d.get("fitted", True)
        return fb
