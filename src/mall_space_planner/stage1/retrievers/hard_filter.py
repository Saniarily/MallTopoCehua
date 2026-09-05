"""Hard-constraint candidate filtering with relaxation diagnostics.

Reproduces the *business rule* used in the legacy system (same ``city_cluster`` and same
``total_area`` bin, thresholds 200 000 / 450 000 m²) as an explicit, configurable and
explainable step. Relaxation order is configurable; every applied relaxation is returned
so that the explainer can report it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mall_space_planner.schemas import PlanningCondition


def area_bin(total_area: float | None, thresholds: list[float]) -> int:
    if total_area is None or pd.isna(total_area):
        return -1
    for i, t in enumerate(thresholds):
        if total_area < t:
            return i
    return len(thresholds)


@dataclass
class FilterResult:
    candidates: pd.DataFrame
    applied: list[str] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)


class HardConstraintFilter:
    """Filter candidate cases by categorical/binned equality constraints."""

    def __init__(
        self,
        area_thresholds: list[float] | None = None,
        use_city_cluster: bool = True,
        use_area_bin: bool = True,
        min_candidates: int = 5,
        allow_relaxation: bool = True,
        city_cluster_col: str = "city_cluster",
        total_area_col: str = "total_area",
        layout_col: str = "layout_type",
    ) -> None:
        self.area_thresholds = list(area_thresholds or [200_000, 450_000])
        self.use_city_cluster = use_city_cluster
        self.use_area_bin = use_area_bin
        self.min_candidates = min_candidates
        self.allow_relaxation = allow_relaxation
        self.city_cluster_col = city_cluster_col
        self.total_area_col = total_area_col
        self.layout_col = layout_col

    def apply(self, query: PlanningCondition, candidates: pd.DataFrame) -> FilterResult:
        res = FilterResult(candidates=candidates)
        cand = candidates
        q_bin = area_bin(query.total_area, self.area_thresholds)

        if query.preferred_layout is not None and self.layout_col in cand:
            sub = cand[cand[self.layout_col].astype(str) == query.preferred_layout.value]
            if len(sub) > 0:
                cand = sub  # user's design decision: enforced whenever any case exists
                res.applied.append(f"{self.layout_col}=={query.preferred_layout.value}")
                if len(sub) < self.min_candidates:
                    res.relaxed.append(f"only {len(sub)} cases of type {query.preferred_layout.value}; other constraints may be relaxed")
            else:
                res.relaxed.append(f"{self.layout_col} preference dropped (no cases of that type)")

        if self.use_city_cluster and query.city_cluster is not None and self.city_cluster_col in cand:
            sub = cand[cand[self.city_cluster_col] == query.city_cluster]
            if len(sub) >= self.min_candidates or not self.allow_relaxation:
                cand = sub
                res.applied.append(f"{self.city_cluster_col}=={query.city_cluster}")
            else:
                res.relaxed.append(f"{self.city_cluster_col} constraint relaxed (only {len(sub)} candidates)")

        if self.use_area_bin and q_bin >= 0 and self.total_area_col in cand:
            bins = cand[self.total_area_col].apply(lambda x: area_bin(x, self.area_thresholds))
            sub = cand[bins == q_bin]
            if len(sub) >= self.min_candidates or not self.allow_relaxation:
                cand = sub
                res.applied.append(f"area_bin=={q_bin} (thresholds={self.area_thresholds})")
            else:
                adj = cand[bins.isin([q_bin - 1, q_bin, q_bin + 1])]
                if len(adj) >= self.min_candidates:
                    cand = adj
                    res.relaxed.append(f"area_bin relaxed to adjacent bins around {q_bin}")
                else:
                    res.relaxed.append("area_bin constraint dropped")

        res.candidates = cand.reset_index(drop=True)
        return res
