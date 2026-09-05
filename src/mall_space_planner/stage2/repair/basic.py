"""Hard-constraint repair for generated layouts.

Fixes, in order: (1) disconnected topology → connect components via the closest node pair;
(2) skeleton nodes outside the boundary → pull towards the polygon's representative point;
(3) shops below the minimum area → merge into an adjacent shop when possible, else drop;
(4) unreachable shops → drop (reported in diagnostics). Each action is logged in
``diagnostics["repairs"]`` so the UI can show what was changed.
"""

from __future__ import annotations

import networkx as nx
from shapely.geometry import Point, Polygon

from mall_space_planner.geometry.ops import boundary_polygon
from mall_space_planner.registry import register
from mall_space_planner.schemas import GeneratedLayout
from mall_space_planner.stage2.base import BaseRepairer, GenerationRequest
from mall_space_planner.topology.convert import from_networkx, to_networkx


@register("repairer", "basic")
class BasicRepairer(BaseRepairer):
    def __init__(self, drop_unreachable: bool = True, merge_small: bool = True) -> None:
        self.drop_unreachable = drop_unreachable
        self.merge_small = merge_small

    def repair(self, layout: GeneratedLayout, request: GenerationRequest) -> GeneratedLayout:
        log: list[str] = []
        g = to_networkx(layout.topology)
        pos = dict(layout.skeleton_positions)
        # (1) connectivity
        comps = list(nx.connected_components(g))
        while len(comps) > 1:
            a, b = comps[0], comps[1]
            best = min(((u, v) for u in a for v in b if u in pos and v in pos), key=lambda e: (pos[e[0]][0] - pos[e[1]][0]) ** 2 + (pos[e[0]][1] - pos[e[1]][1]) ** 2, default=None)
            if best is None:
                best = (next(iter(a)), next(iter(b)))
            g.add_edge(*best)
            log.append(f"connected components via edge {best[0]}-{best[1]}")
            comps = list(nx.connected_components(g))
        # (2) boundary
        poly: Polygon = boundary_polygon(layout.boundary)
        c = poly.representative_point()
        for n, (x, y) in list(pos.items()):
            k = 0
            while not poly.contains(Point(x, y)) and k < 60:
                x, y = x + (c.x - x) * 0.2, y + (c.y - y) * 0.2
                k += 1
            if k:
                pos[n] = (x, y)
                log.append(f"moved node {n} inside boundary")
        # (3)/(4) shops
        units = []
        dropped_small = dropped_unreach = 0
        min_area = request.constraints.shop_area_min or 0.0
        for u in layout.units:
            if u.kind in ("shop", "anchor"):
                if self.drop_unreachable and not u.attrs.get("reachable", True):
                    dropped_unreach += 1
                    continue
                if u.kind == "shop" and min_area and (u.area or 0) < min_area and self.merge_small:
                    # try merging into a touching shop
                    target = None
                    for v in units:
                        if v.kind == "shop" and v.polygon and u.polygon and Polygon(v.polygon).touches(Polygon(u.polygon)):
                            target = v
                            break
                    if target is not None:
                        merged = Polygon(target.polygon).union(Polygon(u.polygon)).buffer(0)
                        if merged.geom_type == "Polygon":
                            target.polygon = [tuple(map(float, p)) for p in merged.exterior.coords[:-1]]
                            target.area = float(merged.area)
                            target.attrs["merged"] = target.attrs.get("merged", 0) + 1
                            log.append(f"merged {u.unit_id} into {target.unit_id}")
                            continue
                    dropped_small += 1
                    continue
            units.append(u)
        new_topo = from_networkx(g)
        new_topo.node_types = layout.topology.node_types
        out = layout.model_copy(update={"topology": new_topo, "skeleton_positions": pos, "units": units})
        out.diagnostics = dict(layout.diagnostics, repairs=log, n_dropped_small=dropped_small, n_dropped_unreachable=dropped_unreach, n_shops=sum(1 for u in units if u.kind == "shop"))
        return out
