"""Geometry decoder v3 (``corridor_only``): Stage-3 fitter + corridor renderer, no shop partition.

The designer gets a complete corridor system (main / secondary widths, loop atria as hints, entrances to
the façade) inside the given boundary; the remaining floor area is intentionally left open.
"""

from __future__ import annotations

import uuid

from shapely.geometry import Polygon

from mall_space_planner.registry import register
from mall_space_planner.schemas import GeneratedLayout, TopologyGraph
from mall_space_planner.stage2.base import BaseGeometryDecoder, GenerationRequest
from mall_space_planner.stage3.fit import CorridorFitter, FitParams
from mall_space_planner.stage3.outline import Outline
from mall_space_planner.stage3.render import RenderParams, render_corridors


@register("geometry_decoder", "corridor_only")
class CorridorOnlyDecoder(BaseGeometryDecoder):
    def __init__(self, corridor_ratio: float = 0.18, main_width: float = 8.0, n_restarts: int = 4, iters: int = 120, max_atria: int = 3) -> None:
        self.render = RenderParams(corridor_ratio=corridor_ratio, main_width=main_width)
        self.n_restarts, self.iters = n_restarts, iters

    def decode(self, topology: TopologyGraph, request: GenerationRequest, seed: int) -> GeneratedLayout:
        c = request.constraints
        outline = Outline(polygon=Polygon(request.boundary.exterior, holes=request.boundary.holes or None).buffer(0), flip_y=False, scale_source="given", source="request")
        res = CorridorFitter(FitParams(shop_depth=c.shop_depth * 0.5, n_restarts=self.n_restarts, iters=self.iters)).fit(topology, outline, seed=seed, skeleton_nodes=set(request.prototype.graph.nodes))
        plan = render_corridors(topology, res.positions, outline, res.roles, self.render, skeleton_nodes=set(request.prototype.graph.nodes))
        diag = {**res.diagnostics, **plan.diagnostics, "inside_ratio": res.diagnostics.get("inside_ratio", 1.0), "site_area_m2": float(outline.area), "n_shops": 0}
        diag.pop("roles", None)
        return GeneratedLayout(layout_id=f"L{uuid.uuid4().hex[:8]}", prototype_id=request.prototype.prototype_id, boundary=request.boundary, topology=topology, skeleton_positions=res.positions, units=plan.units, constraints=c, diagnostics=diag, seed=seed)
