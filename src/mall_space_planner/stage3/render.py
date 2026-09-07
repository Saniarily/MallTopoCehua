"""Turn fitted M positions into a complete corridor system (the Stage-3 "one-click corridor").

* **Hierarchy**: edge *betweenness centrality* on the M graph (a proxy for pedestrian flow) splits edges
  into main (top ``main_quantile``) and secondary corridors; loops of the 2-core are always main.
* **Widths**: ``main_width`` / ``secondary_width`` metres (defaults 8 / 5 – within the 4–9 m range measured
  on the real L polygons of ``B000A0E928_1``; see ``docs/methodology.md`` §Stage 3).
* **Entrances**: from every leaf (dead-end) node and from outer-loop nodes closest to the façade, extend
  a short corridor stub perpendicular to the façade until it meets the outline (max ``n_entrances``, at
  least ``min_entrances``, no two closer than ``entrance_spacing``).
* **Atria**: holes enclosed by main-corridor loops, shrunk so that an inward-facing shop ring remains
  (cap ``atrium_area_max``) – optional visual hint, not a hard element.
* **Junction pads**: square pads at M nodes with degree ≥ 3 so crossings read as plazas.

Output ``CorridorPlan`` carries shapely polygons (metres) + a ``GeneratedLayout``-compatible unit list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from mall_space_planner.schemas import SpaceUnit, TopologyGraph
from mall_space_planner.stage3.outline import Outline
from mall_space_planner.topology.convert import to_networkx


@dataclass
class RenderParams:
    main_width: float = 8.0  # upper bound; the actual width is derived from ``corridor_ratio`` (see ``widths_for``)
    secondary_width: float = 5.0
    corridor_ratio: float = 0.18  # target corridor area / floor area (real malls ≈ 0.12–0.25)
    min_width: float = 3.0
    secondary_factor: float = 0.65  # secondary width = factor × main width
    main_quantile: float = 0.6  # edges above this betweenness quantile are "main"
    min_entrances: int = 2
    max_entrances: int = 6
    entrance_spacing: float = 30.0
    entrance_width: float = 6.0
    atrium_area_max: float = 900.0
    min_atrium_area: float = 80.0
    junction_pad: float = 1.25  # pad side = width × this


@dataclass
class CorridorPlan:
    corridors_main: Polygon | MultiPolygon
    corridors_secondary: Polygon | MultiPolygon
    entrances: list[dict]  # {node, point, stub: Polygon}
    atria: list[Polygon]
    edge_class: dict[tuple[str, str], str]
    units: list[SpaceUnit] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def corridor_union(self):  # noqa: ANN201
        parts = [self.corridors_main, self.corridors_secondary] + [e["stub"] for e in self.entrances]
        return unary_union([p for p in parts if p is not None and not p.is_empty])


def _polys(geom) -> list[Polygon]:  # noqa: ANN001
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def classify_edges(g: nx.Graph, quantile: float, roles: dict[str, str] | None = None) -> dict[tuple[str, str], str]:
    """main = high edge-betweenness (top ``1-quantile``) or an edge of the outer loop; else secondary."""
    bc = nx.edge_betweenness_centrality(g) if g.number_of_edges() else {}
    vals = np.array(list(bc.values())) if bc else np.array([0.0])
    thr = float(np.quantile(vals, quantile)) if len(vals) else 0.0
    out = {}
    for u, v in g.edges:
        b = bc.get((u, v), bc.get((v, u), 0.0))
        on_outer = bool(roles) and roles.get(u) == "outer" and roles.get(v) == "outer"
        out[(u, v)] = "main" if (on_outer or b >= thr) else "secondary"
    return out


def widths_for(site_area: float, main_len: float, sec_len: float, prm: RenderParams) -> tuple[float, float]:
    """Corridor widths such that corridor area ≈ ``corridor_ratio`` × floor area (clipped to [min, max])."""
    budget = prm.corridor_ratio * site_area
    denom = main_len + prm.secondary_factor * sec_len
    w = budget / denom if denom > 1e-9 else prm.main_width
    w = float(np.clip(w, prm.min_width, prm.main_width))
    return w, float(max(prm.min_width, w * prm.secondary_factor))


def _stub_to_facade(p: np.ndarray, outline: Polygon, direction_hint: np.ndarray | None, width: float) -> tuple[Polygon | None, np.ndarray | None]:
    """Shortest perpendicular-ish stub from ``p`` to the façade. Returns (stub polygon, façade point)."""
    ext = outline.exterior
    q = ext.interpolate(ext.project(Point(p)))
    fp = np.array([q.x, q.y])
    d = fp - p
    L = np.linalg.norm(d)
    if L < 1e-6:
        return None, fp
    # prefer the hinted direction (continuation of the dead-end corridor) if it reaches the façade quickly
    if direction_hint is not None and np.linalg.norm(direction_hint) > 1e-9:
        h = direction_hint / np.linalg.norm(direction_hint)
        ray = LineString([p, p + h * (L * 3 + 50)])
        hit = ray.intersection(ext)
        pts = [np.array(pt.coords[0]) for pt in getattr(hit, "geoms", [hit]) if not pt.is_empty and pt.geom_type == "Point"]
        if pts:
            cand = min(pts, key=lambda x: np.linalg.norm(x - p))
            if np.linalg.norm(cand - p) < 2.0 * L + 5:
                fp = cand
    stub = LineString([p, fp]).buffer(width / 2, cap_style="flat").buffer(width * 0.02)
    return stub, fp


def render_corridors(topology: TopologyGraph, positions: dict[str, tuple[float, float]], outline: Outline, roles: dict[str, str] | None = None, params: RenderParams | None = None) -> CorridorPlan:
    prm = params or RenderParams()
    g = to_networkx(topology)
    pos = {k: np.asarray(v, float) for k, v in positions.items()}
    site = outline.polygon
    cls = classify_edges(g, prm.main_quantile, roles)
    main_segs, sec_segs = [], []
    for (u, v), c in cls.items():
        if u in pos and v in pos and np.linalg.norm(pos[u] - pos[v]) > 1e-6:
            (main_segs if c == "main" else sec_segs).append(LineString([pos[u], pos[v]]))
    w_main, w_sec = widths_for(site.area, sum(s.length for s in main_segs), sum(s.length for s in sec_segs), prm)
    main = unary_union([s.buffer(w_main / 2, cap_style="flat", join_style="mitre") for s in main_segs]) if main_segs else Polygon()
    sec = unary_union([s.buffer(w_sec / 2, cap_style="flat", join_style="mitre") for s in sec_segs]) if sec_segs else Polygon()
    # junction pads
    pads = []
    for v in g.nodes:
        if g.degree(v) >= 3 and v in pos:
            w = w_main if any(cls.get((v, u), cls.get((u, v))) == "main" for u in g.neighbors(v)) else w_sec
            s = w * prm.junction_pad / 2
            pads.append(Polygon([pos[v] + [-s, -s], pos[v] + [s, -s], pos[v] + [s, s], pos[v] + [-s, s]]))
    if pads:
        main = unary_union([main, *pads])
    main = main.intersection(site).buffer(0)
    sec = sec.difference(main).intersection(site).buffer(0)
    # entrances
    entrances: list[dict] = []
    cands = []
    for v in g.nodes:
        if v not in pos:
            continue
        dist = site.exterior.distance(Point(pos[v]))
        if g.degree(v) == 1:  # dead-end corridors end at an entrance / anchor
            u = next(iter(g.neighbors(v)))
            cands.append((0, dist, v, pos[v] - pos[u]))
        elif roles and roles.get(v) == "outer":
            cands.append((1, dist, v, None))
    cands.sort(key=lambda t: (t[0], t[1]))
    for pri, dist, v, hint in cands:
        if len(entrances) >= prm.max_entrances:
            break
        stub, fp = _stub_to_facade(pos[v], site, hint, min(prm.entrance_width, w_main))
        if fp is None:
            continue
        if any(np.linalg.norm(fp - e["point"]) < prm.entrance_spacing for e in entrances):
            continue
        if pri == 1 and len(entrances) >= prm.min_entrances and dist > 2.5 * w_main:
            continue  # outer-loop candidates only to reach the minimum, unless they touch the façade anyway
        entrances.append({"node": v, "point": fp, "stub": (stub.intersection(site).buffer(0) if stub is not None else Polygon()), "kind": "dead_end" if pri == 0 else "loop"})
    # atria = holes enclosed by main corridors (centre-line faces), shrunk to leave an inward shop ring
    atria: list[Polygon] = []
    if main_segs:
        faces = list(polygonize(unary_union(main_segs)))
        for f in sorted(faces, key=lambda q: -q.area):
            hole = f.buffer(-w_main / 2).buffer(0)
            if hole.is_empty or hole.area < prm.min_atrium_area:
                continue
            if hole.area > prm.atrium_area_max:
                lo, hi = 0.0, float(np.sqrt(hole.area))
                for _ in range(25):
                    mid = (lo + hi) / 2
                    h2 = hole.buffer(-mid)
                    if h2.is_empty or h2.area < prm.atrium_area_max:
                        hi = mid
                    else:
                        lo = mid
                h2 = hole.buffer(-lo)
                hole = max(_polys(h2), key=lambda q: q.area) if _polys(h2) else hole
            atria.append(hole)
    # units (for GeneratedLayout / exports)
    units: list[SpaceUnit] = []
    for i, p in enumerate(_polys(main)):
        units.append(SpaceUnit(unit_id=f"CM{i}", kind="corridor", polygon=[tuple(map(float, c)) for c in p.exterior.coords[:-1]], centroid=(p.centroid.x, p.centroid.y), area=float(p.area), attrs={"class": "main", "width": w_main}))
    for i, p in enumerate(_polys(sec)):
        units.append(SpaceUnit(unit_id=f"CS{i}", kind="corridor", polygon=[tuple(map(float, c)) for c in p.exterior.coords[:-1]], centroid=(p.centroid.x, p.centroid.y), area=float(p.area), attrs={"class": "secondary", "width": w_sec}))
    for i, a in enumerate(atria):
        units.append(SpaceUnit(unit_id=f"AT{i}", kind="atrium", polygon=[tuple(map(float, c)) for c in a.exterior.coords[:-1]], centroid=(a.centroid.x, a.centroid.y), area=float(a.area)))
    for i, e in enumerate(entrances):
        units.append(SpaceUnit(unit_id=f"E{i}", kind="entrance", centroid=(float(e["point"][0]), float(e["point"][1])), attached_to=[e["node"]], polygon=[tuple(map(float, c)) for c in e["stub"].exterior.coords[:-1]] if not e["stub"].is_empty else None, attrs={"kind": e["kind"]}))
    corr = unary_union([main, sec, *[e["stub"] for e in entrances if not e["stub"].is_empty]])
    diag = {
        "main_width_m": w_main, "secondary_width_m": w_sec,
        "n_main_edges": sum(1 for c in cls.values() if c == "main"), "n_secondary_edges": sum(1 for c in cls.values() if c == "secondary"),
        "corridor_area_m2": float(corr.area), "corridor_ratio": float(corr.area / max(site.area, 1e-9)), "n_entrances": len(entrances), "n_atria": len(atria),
        "atrium_area_m2": float(sum(a.area for a in atria)), "main_length_m": float(sum(s.length for s in main_segs)), "secondary_length_m": float(sum(s.length for s in sec_segs)),
    }
    return CorridorPlan(corridors_main=main, corridors_secondary=sec, entrances=entrances, atria=atria, edge_class=cls, units=units, diagnostics=diag)


__all__ = ["CorridorPlan", "RenderParams", "classify_edges", "render_corridors"]
