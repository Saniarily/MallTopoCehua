"""Computational-geometry helpers (shapely-based) and export formats."""

from mall_space_planner.geometry.export import layout_to_geojson, layout_to_json, layout_to_svg
from mall_space_planner.geometry.ops import (
    boundary_polygon,
    fit_positions_to_boundary,
    inside_ratio,
    layout_positions,
    normalize_positions,
)

__all__ = [
    "layout_to_geojson",
    "layout_to_json",
    "layout_to_svg",
    "boundary_polygon",
    "fit_positions_to_boundary",
    "inside_ratio",
    "layout_positions",
    "normalize_positions",
]
