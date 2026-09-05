"""Geometric evaluator for generated layouts (boundary, overlap, validity, reachability, areas, constraints)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from mall_space_planner.geometry.ops import boundary_polygon
from mall_space_planner.registry import register
from mall_space_planner.schemas import ConstraintSet, EvaluationResult, GeneratedLayout


@register("evaluator", "geometry")
class GeometryEvaluator:
    def __init__(self, overlap_tol: float = 1e-6, max_overlap_rate: float = 0.01, min_inside_ratio: float = 0.98, min_reachable_rate: float = 0.9) -> None:
        self.overlap_tol = overlap_tol
        self.max_overlap_rate = max_overlap_rate
        self.min_inside_ratio = min_inside_ratio
        self.min_reachable_rate = min_reachable_rate

    def evaluate(self, layout: GeneratedLayout, constraints: ConstraintSet | None = None) -> EvaluationResult:
        c = constraints or layout.constraints or ConstraintSet()
        site = boundary_polygon(layout.boundary)
        polys = [(u, Polygon(u.polygon)) for u in layout.units if u.polygon]
        shops = [(u, p) for u, p in polys if u.kind == "shop"]
        n_invalid = sum(1 for _, p in polys if not p.is_valid or p.is_empty)
        total_area = sum(p.area for _, p in polys) or 1.0
        inside_area = sum(p.intersection(site).area for _, p in polys)
        overlap_area = 0.0
        for (_, a), (_, b) in combinations([sp for sp in shops], 2):
            if a.intersects(b):
                overlap_area += a.intersection(b).area
        shop_area = sum(p.area for _, p in shops) or 1.0
        nodes_inside = float(np.mean([site.contains(Point(p)) or site.touches(Point(p)) for p in layout.skeleton_positions.values()])) if layout.skeleton_positions else 0.0
        reach = [bool(u.attrs.get("reachable", True)) for u, _ in shops]
        areas = np.array([p.area for _, p in shops]) if shops else np.array([])
        used = unary_union([p for _, p in polys]).area if polys else 0.0

        metrics: dict = {
            "n_shops": len(shops),
            "inside_area_ratio": inside_area / total_area,
            "node_inside_ratio": nodes_inside,
            "shop_overlap_rate": overlap_area / shop_area,
            "invalid_polygon_rate": n_invalid / max(1, len(polys)),
            "shop_reachable_rate": float(np.mean(reach)) if reach else 1.0,
            "site_coverage": used / site.area if site.area else 0.0,
            "shop_area_mean": float(areas.mean()) if areas.size else 0.0,
            "shop_area_std": float(areas.std()) if areas.size else 0.0,
            "shop_area_min": float(areas.min()) if areas.size else 0.0,
            "shop_area_max": float(areas.max()) if areas.size else 0.0,
        }
        checks: dict[str, bool] = {
            "inside_boundary": metrics["inside_area_ratio"] >= self.min_inside_ratio and nodes_inside >= self.min_inside_ratio,
            "no_overlap": metrics["shop_overlap_rate"] <= self.max_overlap_rate,
            "valid_polygons": n_invalid == 0,
            "shops_reachable": metrics["shop_reachable_rate"] >= self.min_reachable_rate,
        }
        # soft constraint satisfaction
        soft: dict[str, bool] = {}
        if c.target_num_shops:
            metrics["shop_count_error_pct"] = abs(len(shops) - c.target_num_shops) / c.target_num_shops * 100
            soft["shop_count"] = metrics["shop_count_error_pct"] <= 30
        if c.shop_area_min and areas.size:
            metrics["shops_below_min_rate"] = float((areas < c.shop_area_min).mean())
            soft["shop_area_min"] = metrics["shops_below_min_rate"] <= 0.1
        if c.shop_area_max and areas.size:
            metrics["shops_above_max_rate"] = float((areas > c.shop_area_max * 1.05).mean())
            soft["shop_area_max"] = metrics["shops_above_max_rate"] <= 0.1
        if c.shop_area_mean and areas.size:
            metrics["shop_area_mean_error_pct"] = abs(areas.mean() - c.shop_area_mean) / c.shop_area_mean * 100
            soft["shop_area_mean"] = metrics["shop_area_mean_error_pct"] <= 30
        n_entr = sum(1 for u in layout.units if u.kind == "entrance")
        metrics["n_entrances"] = n_entr
        soft["entrances"] = n_entr >= c.min_entrances
        metrics["constraint_satisfaction_rate"] = float(np.mean(list(soft.values()))) if soft else 1.0
        passed = {**checks, **{f"soft_{k}": v for k, v in soft.items()}}
        return EvaluationResult(evaluator="geometry", metrics=metrics, passed=passed, overall_pass=all(checks.values()), details={"hard_checks": checks, "soft_checks": soft})
