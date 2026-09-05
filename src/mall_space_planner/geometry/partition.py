"""Shop partitioning: carve the site polygon minus the corridor network into shop units.

Algorithm (rule-based, deterministic given a seed):

1. **Corridor polygon** = union of buffered corridor segments (skeleton edges) with width
   ``corridor_width``; atria = circular buffers at chosen junction nodes.
2. **Leasable region** = site ∖ (corridors ∪ atria), split into its connected polygons.
3. Each leasable polygon is **recursively split** (guillotine cuts alternating along the
   longer axis) until pieces fall into the target area window. Cuts are placed at the
   longer-axis midpoint with small jitter so that different seeds give different but valid
   partitions.
4. Every unit is **attached** to the nearest corridor segment (frontage). Units within
   ``max_frontage_dist`` of a corridor are *reachable shops*; deeper units (up to
   ``shop_depth``) are kept as ``anchor`` units (department store / supermarket-like
   deep spaces reached through the frontage); units deeper than ``shop_depth`` are flagged
   ``unreachable`` and left to the repairer.

All outputs are valid shapely polygons clipped to the site, so overlap/out-of-bounds are
impossible by construction; the evaluator still checks them independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from mall_space_planner.schemas import SpaceUnit, TopologyGraph


@dataclass
class PartitionParams:
    corridor_width: float = 6.0
    atrium_radius: float = 9.0
    shop_area_min: float = 60.0
    shop_area_max: float = 400.0
    min_piece_area: float = 25.0
    max_frontage_dist: float = 4.0
    shop_depth: float = 30.0
    anchor_area_max: float = 2500.0
    jitter: float = 0.12


def corridor_polygon(topology: TopologyGraph, pos: dict[str, tuple[float, float]], width: float) -> Polygon | MultiPolygon:
    segs = [LineString([pos[u], pos[v]]) for u, v in topology.edges() if u in pos and v in pos]
    if not segs:
        return Polygon()
    return unary_union([s.buffer(width / 2.0, cap_style="flat", join_style="mitre") for s in segs]).buffer(0)


def pick_atria(topology: TopologyGraph, pos: dict[str, tuple[float, float]], n: int) -> list[str]:
    """Atria at the highest-degree junctions (ties broken by centrality to the site centre)."""
    if n <= 0:
        return []
    deg = topology.degree()
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    cx, cy = xs.mean(), ys.mean()
    order = sorted(pos, key=lambda k: (-deg.get(k, 0), (pos[k][0] - cx) ** 2 + (pos[k][1] - cy) ** 2))
    return order[:n]


def _split_polygon(poly: Polygon, area_max: float, min_piece: float, rng: np.random.RandomState, jitter: float, depth: int = 0) -> list[Polygon]:
    if poly.is_empty or poly.area < min_piece:
        return []
    if poly.area <= area_max or depth > 12:
        return [poly]
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    frac = 0.5 + rng.uniform(-jitter, jitter)
    if w >= h:
        cut = minx + w * frac
        a = Polygon([(minx - 1, miny - 1), (cut, miny - 1), (cut, maxy + 1), (minx - 1, maxy + 1)])
        b = Polygon([(cut, miny - 1), (maxx + 1, miny - 1), (maxx + 1, maxy + 1), (cut, maxy + 1)])
    else:
        cut = miny + h * frac
        a = Polygon([(minx - 1, miny - 1), (maxx + 1, miny - 1), (maxx + 1, cut), (minx - 1, cut)])
        b = Polygon([(minx - 1, cut), (maxx + 1, cut), (maxx + 1, maxy + 1), (minx - 1, maxy + 1)])
    out: list[Polygon] = []
    for half in (poly.intersection(a), poly.intersection(b)):
        for piece in _iter_polys(half):
            out.extend(_split_polygon(piece, area_max, min_piece, rng, jitter, depth + 1))
    return out


def _iter_polys(geom) -> list[Polygon]:  # noqa: ANN001
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def partition_shops(
    site: Polygon,
    topology: TopologyGraph,
    pos: dict[str, tuple[float, float]],
    params: PartitionParams,
    n_atria: int,
    seed: int,
) -> tuple[list[SpaceUnit], dict[str, float]]:
    """Return space units (corridor, atria, shops) and partition diagnostics."""
    rng = np.random.RandomState(seed)
    corridors = corridor_polygon(topology, pos, params.corridor_width).intersection(site)
    atria_nodes = pick_atria(topology, pos, n_atria)
    atria = [Point(pos[n]).buffer(params.atrium_radius).intersection(site) for n in atria_nodes]
    blocked = unary_union([corridors, *atria]) if atria else corridors
    leasable = site.difference(blocked).buffer(0)

    units: list[SpaceUnit] = []
    for i, g in enumerate(_iter_polys(corridors)):
        units.append(SpaceUnit(unit_id=f"C{i}", kind="corridor", polygon=[tuple(map(float, p)) for p in g.exterior.coords[:-1]], centroid=(g.centroid.x, g.centroid.y), area=float(g.area)))
    for n, g in zip(atria_nodes, atria, strict=True):
        for gg in _iter_polys(g):
            units.append(SpaceUnit(unit_id=f"AT_{n}", kind="atrium", polygon=[tuple(map(float, p)) for p in gg.exterior.coords[:-1]], centroid=(gg.centroid.x, gg.centroid.y), area=float(gg.area), attached_to=[n]))

    segs = {(u, v): LineString([pos[u], pos[v]]) for u, v in topology.edges() if u in pos and v in pos}
    n_unreach = 0
    shop_i = anchor_i = 0
    corridor_zone = blocked.buffer(params.max_frontage_dist)
    for region in _iter_polys(leasable):
        # Frontage band (near corridors) is cut into shops; the remainder into larger anchor units.
        front = region.intersection(corridor_zone.buffer(params.shop_depth * 0.5)).buffer(0)
        deep = region.difference(front).buffer(0)
        pieces = [(p, "shop") for f in _iter_polys(front) for p in _split_polygon(f, params.shop_area_max, params.min_piece_area, rng, params.jitter)]
        pieces += [(p, "anchor") for d in _iter_polys(deep) for p in _split_polygon(d, params.anchor_area_max, params.min_piece_area, rng, params.jitter)]
        for piece, kind in pieces:
            piece = piece.buffer(0)
            if piece.is_empty or piece.area < params.min_piece_area:
                continue
            best, dist = None, float("inf")
            for key, seg in segs.items():
                d = piece.distance(seg) - params.corridor_width / 2.0
                if d < dist:
                    best, dist = key, d
            reachable = dist <= (params.max_frontage_dist if kind == "shop" else params.shop_depth)
            n_unreach += int(not reachable)
            uid = f"S{shop_i}" if kind == "shop" else f"A{anchor_i}"
            if kind == "shop":
                shop_i += 1
            else:
                anchor_i += 1
            units.append(
                SpaceUnit(
                    unit_id=uid,
                    kind=kind,
                    polygon=[tuple(map(float, p)) for p in piece.exterior.coords[:-1]],
                    centroid=(piece.centroid.x, piece.centroid.y),
                    area=float(piece.area),
                    attached_to=list(best) if best else [],
                    attrs={"frontage_dist": float(max(0.0, dist)), "reachable": reachable, "too_small": kind == "shop" and piece.area < params.shop_area_min},
                )
            )
    shops = [u for u in units if u.kind == "shop"]
    diag = {
        "corridor_area": float(corridors.area),
        "atrium_area": float(sum(a.area for a in atria)),
        "leasable_area": float(leasable.area),
        "n_shops": len(shops),
        "n_anchors": sum(1 for u in units if u.kind == "anchor"),
        "n_unreachable_units": n_unreach,
        "n_shops_below_min": sum(1 for u in shops if u.attrs.get("too_small")),
        "shop_area_mean": float(np.mean([u.area for u in shops])) if shops else 0.0,
    }
    return units, diag
