"""Minimal geometry decoder: embed the expanded topology inside the site boundary."""

from __future__ import annotations

import uuid

from mall_space_planner.geometry.ops import fit_positions_to_boundary, inside_ratio, layout_positions
from mall_space_planner.registry import register
from mall_space_planner.schemas import GeneratedLayout, TopologyGraph
from mall_space_planner.stage2.base import BaseGeometryDecoder, GenerationRequest


@register("geometry_decoder", "skeleton_embed")
class SkeletonEmbedDecoder(BaseGeometryDecoder):
    def __init__(self, margin_frac: float = 0.08) -> None:
        self.margin_frac = margin_frac

    def decode(self, topology: TopologyGraph, request: GenerationRequest, seed: int) -> GeneratedLayout:
        pos01 = layout_positions(topology, seed=seed)
        pos = fit_positions_to_boundary(pos01, request.boundary, self.margin_frac)
        return GeneratedLayout(
            layout_id=f"L{uuid.uuid4().hex[:8]}",
            prototype_id=request.prototype.prototype_id,
            boundary=request.boundary,
            topology=topology,
            skeleton_positions=pos,
            constraints=request.constraints,
            diagnostics={"inside_ratio": inside_ratio(pos, request.boundary), "site_area_m2": request.boundary.area()},
            seed=seed,
        )
