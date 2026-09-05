"""Full geometry decoder: skeleton embedding → corridor buffers → atria → shop partition."""

from __future__ import annotations

import uuid

from shapely.geometry import Point

from mall_space_planner.geometry.ops import boundary_polygon, fit_positions_to_boundary, inside_ratio, layout_positions
from mall_space_planner.geometry.partition import PartitionParams, partition_shops
from mall_space_planner.registry import register
from mall_space_planner.schemas import GeneratedLayout, SpaceUnit, TopologyGraph
from mall_space_planner.stage2.base import BaseGeometryDecoder, GenerationRequest


@register("geometry_decoder", "corridor_partition")
class CorridorPartitionDecoder(BaseGeometryDecoder):
    def __init__(self, margin_frac: float = 0.1, atrium_radius: float = 9.0, min_piece_area: float = 25.0, max_frontage_dist: float = 4.0, default_shop_area: tuple[float, float] = (60.0, 400.0)) -> None:
        self.margin_frac = margin_frac
        self.atrium_radius = atrium_radius
        self.min_piece_area = min_piece_area
        self.max_frontage_dist = max_frontage_dist
        self.default_shop_area = default_shop_area

    def decode(self, topology: TopologyGraph, request: GenerationRequest, seed: int) -> GeneratedLayout:
        c = request.constraints
        pos = fit_positions_to_boundary(layout_positions(topology, seed=seed), request.boundary, self.margin_frac)
        site = boundary_polygon(request.boundary)
        params = PartitionParams(
            corridor_width=c.corridor_width,
            atrium_radius=self.atrium_radius,
            shop_area_min=c.shop_area_min or self.default_shop_area[0],
            shop_area_max=c.shop_area_max or self.default_shop_area[1],
            min_piece_area=self.min_piece_area,
            max_frontage_dist=self.max_frontage_dist,
            shop_depth=c.shop_depth,
        )
        n_atria = c.num_atria if c.num_atria is not None else (1 if topology.num_nodes >= 8 else 0)
        units, diag = partition_shops(site, topology, pos, params, n_atria, seed)
        # Entrances: user-marked points snapped to the nearest skeleton node; else the node nearest to the boundary.
        entrances = []
        if request.boundary.entrances:
            for i, e in enumerate(request.boundary.entrances):
                nearest = min(pos, key=lambda k: (pos[k][0] - e[0]) ** 2 + (pos[k][1] - e[1]) ** 2)
                entrances.append((f"E{i}", e, nearest))
        else:
            ext = site.exterior
            for i in range(max(1, c.min_entrances)):
                cand = sorted(pos, key=lambda k: ext.distance(Point(pos[k])))
                node = cand[min(i, len(cand) - 1)]
                p = ext.interpolate(ext.project(Point(pos[node])))
                entrances.append((f"E{i}", (p.x, p.y), node))
        for uid, p, node in entrances:
            units.append(SpaceUnit(unit_id=uid, kind="entrance", centroid=(float(p[0]), float(p[1])), attached_to=[node]))
        diag.update({"inside_ratio": inside_ratio(pos, request.boundary), "site_area_m2": float(site.area), "n_entrances": len(entrances)})
        return GeneratedLayout(layout_id=f"L{uuid.uuid4().hex[:8]}", prototype_id=request.prototype.prototype_id, boundary=request.boundary, topology=topology, skeleton_positions=pos, units=units, constraints=c, diagnostics=diag, seed=seed)
