"""Geometry decoder v2: mall-like planar corridor embedding → corridor buffers → loop-hole atria → shops.

Differences from ``corridor_partition`` (v1, force-directed positions + circular atria at junctions):

* positions come from :func:`planar_corridor_embedding` – loops are opened up (Tutte embedding of the
  2-core with the outer cycle pinned to a rounded outline), branches leave the core corridors
  perpendicularly and edge directions are softly snapped to a right-angle frame;
* the abstract drawing is fitted into the site boundary with a *uniform* scale (no anisotropic
  stretching) and the outer corridor loop is pushed towards the boundary so that a shop band of
  ``shop_depth`` fits between corridor and façade;
* atria are the largest holes enclosed by corridor loops (``num_atria`` from the constraints caps the
  count) instead of circles at high-degree nodes.
"""

from __future__ import annotations

import uuid

import numpy as np
from shapely.geometry import Point, Polygon

from mall_space_planner.geometry.ops import boundary_polygon, inside_ratio
from mall_space_planner.geometry.partition import PartitionParams, partition_shops
from mall_space_planner.geometry.planar_embed import PlanarEmbedParams, planar_corridor_embedding
from mall_space_planner.registry import register
from mall_space_planner.schemas import GeneratedLayout, SpaceUnit, TopologyGraph
from mall_space_planner.stage2.base import BaseGeometryDecoder, GenerationRequest


def fit_uniform(pos: dict[str, tuple[float, float]], site: Polygon, margin: float) -> tuple[dict[str, tuple[float, float]], float, tuple[float, float]]:
    """Uniformly scale + translate abstract positions so their bbox fits inside ``site.bounds`` minus ``margin``.
    Returns (positions, scale, offset). Points that still fall outside the polygon are pulled inward."""
    if not pos:
        return {}, 1.0, (0.0, 0.0)
    P = np.array(list(pos.values()), float)
    lo, hi = P.min(0), P.max(0)
    span = np.maximum(hi - lo, 1e-9)
    minx, miny, maxx, maxy = site.bounds
    avail = np.array([maxx - minx, maxy - miny]) - 2 * margin
    # allow a 90° rotation when the drawing's aspect is opposite to the site's
    rot = (span[0] > span[1]) != (avail[0] > avail[1])
    if rot:
        P = P[:, ::-1] * np.array([1.0, -1.0])
        lo, hi = P.min(0), P.max(0)
        span = np.maximum(hi - lo, 1e-9)
    s = float(min(avail / span))
    centre_site = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    centre_draw = (lo + hi) / 2
    Q = (P - centre_draw) * s + centre_site
    out = {}
    c = site.representative_point()
    for k, (x, y) in zip(pos.keys(), Q, strict=True):
        p = Point(x, y)
        i = 0
        while not site.contains(p) and i < 60:
            x, y = x + (c.x - x) * 0.1, y + (c.y - y) * 0.1
            p = Point(x, y)
            i += 1
        out[k] = (float(x), float(y))
    return out, s, (float(centre_site[0] - centre_draw[0] * s), float(centre_site[1] - centre_draw[1] * s))


@register("geometry_decoder", "planar_corridor")
class PlanarCorridorDecoder(BaseGeometryDecoder):
    def __init__(
        self,
        margin_frac: float = 0.16,
        min_piece_area: float = 25.0,
        max_frontage_dist: float = 4.0,
        default_shop_area: tuple[float, float] = (60.0, 400.0),
        ortho_weight: float = 0.6,
        branch_step: float = 1.0,
        max_atria: int = 3,
        atrium_area_max: float = 900.0,
    ) -> None:
        self.atrium_area_max = atrium_area_max
        self.margin_frac = margin_frac
        self.min_piece_area = min_piece_area
        self.max_frontage_dist = max_frontage_dist
        self.default_shop_area = default_shop_area
        self.embed = PlanarEmbedParams(ortho_weight=ortho_weight, branch_step=branch_step, max_atria=max_atria)

    def decode(self, topology: TopologyGraph, request: GenerationRequest, seed: int) -> GeneratedLayout:
        c = request.constraints
        site = boundary_polygon(request.boundary)
        pos_abs, info = planar_corridor_embedding(topology, self.embed)
        minx, miny, maxx, maxy = site.bounds
        # margin = shop band depth (so shops fit between the outer corridor and the façade), bounded by margin_frac
        margin = min(max(c.shop_depth * 0.6, c.corridor_width), self.margin_frac * min(maxx - minx, maxy - miny))
        pos, s, _ = fit_uniform(pos_abs, site, margin)
        # atrium footprints: transform the abstract face polygons with the same mapping (via nearest-node fit)
        atria_polys: list[Polygon] = []
        if info["faces"] and len(pos_abs) >= 2:
            keys = list(pos_abs)
            A = np.array([pos_abs[k] for k in keys])
            B = np.array([pos[k] for k in keys])
            # solve similarity (allow rotation/reflection) A -> B by least squares on homogeneous coords
            X = np.c_[A, np.ones(len(A))]
            M, *_ = np.linalg.lstsq(X, B, rcond=None)
            for ring in info["faces"]:
                R = np.array(ring)
                Q = np.c_[R, np.ones(len(R))] @ M
                poly = Polygon(Q).buffer(0)
                if poly.is_valid and not poly.is_empty:
                    atria_polys.append(poly)
        n_atria = c.num_atria if c.num_atria is not None else len(atria_polys)
        atria_polys = atria_polys[: max(0, n_atria)]
        params = PartitionParams(
            corridor_width=c.corridor_width,
            shop_area_min=c.shop_area_min or self.default_shop_area[0],
            shop_area_max=c.shop_area_max or self.default_shop_area[1],
            min_piece_area=self.min_piece_area,
            max_frontage_dist=self.max_frontage_dist,
            shop_depth=c.shop_depth,
            atrium_area_max=self.atrium_area_max,
        )
        units, diag = partition_shops(site, topology, pos, params, n_atria, seed, atrium_polygons=atria_polys)
        entrances = []
        if request.boundary.entrances:
            for i, e in enumerate(request.boundary.entrances):
                nearest = min(pos, key=lambda k: (pos[k][0] - e[0]) ** 2 + (pos[k][1] - e[1]) ** 2)
                entrances.append((f"E{i}", e, nearest))
        else:
            ext = site.exterior
            # entrances at the ends of branches / outer-loop nodes closest to the façade
            cand = sorted(pos, key=lambda k: ext.distance(Point(pos[k])))
            for i in range(max(1, c.min_entrances)):
                node = cand[min(i, len(cand) - 1)]
                p = ext.interpolate(ext.project(Point(pos[node])))
                entrances.append((f"E{i}", (p.x, p.y), node))
        for uid, p, node in entrances:
            units.append(SpaceUnit(unit_id=uid, kind="entrance", centroid=(float(p[0]), float(p[1])), attached_to=[node]))
        diag.update({
            "inside_ratio": inside_ratio(pos, request.boundary), "site_area_m2": float(site.area), "n_entrances": len(entrances),
            "embedding_crossings": int(info["crossings"]), "n_core_nodes": len(info["core_nodes"]), "n_loop_atria": len(atria_polys), "draw_scale_m": float(s),
        })
        return GeneratedLayout(layout_id=f"L{uuid.uuid4().hex[:8]}", prototype_id=request.prototype.prototype_id, boundary=request.boundary, topology=topology, skeleton_positions=pos, units=units, constraints=c, diagnostics=diag, seed=seed)
