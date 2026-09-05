"""Boundary-aware placement of skeleton nodes (Phase 3 will add full unit generation)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

from mall_space_planner.schemas import SiteBoundary, TopologyGraph
from mall_space_planner.topology.convert import to_networkx


def boundary_polygon(b: SiteBoundary) -> Polygon:
    return Polygon(b.exterior, holes=b.holes or None)


def normalize_positions(pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Centre and scale positions to the unit square [0,1]^2."""
    if not pos:
        return {}
    arr = np.array(list(pos.values()), dtype=float)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    arr = (arr - lo) / span
    return {k: (float(x), float(y)) for k, (x, y) in zip(pos.keys(), arr, strict=True)}


def layout_positions(graph: TopologyGraph, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Use prototype coordinates when present, otherwise a seeded spring layout."""
    g = to_networkx(graph)
    known = {n: graph.positions[n] for n in graph.nodes if n in graph.positions}
    if len(known) == len(graph.nodes) and known:
        return normalize_positions(known)
    pos = nx.spring_layout(g, pos=known or None, fixed=list(known) or None, seed=seed, iterations=200)
    return normalize_positions({str(k): (float(v[0]), float(v[1])) for k, v in pos.items()})


def fit_positions_to_boundary(
    pos01: dict[str, tuple[float, float]], boundary: SiteBoundary, margin_frac: float = 0.08, max_push: int = 50
) -> dict[str, tuple[float, float]]:
    """Map unit-square positions into the boundary's bounding box and push outside points inward."""
    poly = boundary_polygon(boundary)
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    out: dict[str, tuple[float, float]] = {}
    c = poly.representative_point()
    for k, (x, y) in pos01.items():
        px = minx + (margin_frac + (1 - 2 * margin_frac) * x) * w
        py = miny + (margin_frac + (1 - 2 * margin_frac) * y) * h
        p = Point(px, py)
        i = 0
        while not poly.contains(p) and i < max_push:
            px, py = px + (c.x - px) * 0.15, py + (c.y - py) * 0.15
            p = Point(px, py)
            i += 1
        out[k] = (float(px), float(py))
    return out


def inside_ratio(pos: dict[str, tuple[float, float]], boundary: SiteBoundary) -> float:
    if not pos:
        return 0.0
    poly = boundary_polygon(boundary)
    return float(np.mean([poly.contains(Point(p)) or poly.touches(Point(p)) for p in pos.values()]))
