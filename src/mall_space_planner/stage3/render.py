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
    entrance_reach: float = 4.0  # a dead end within this × main width of the façade becomes an entrance, else a vertical core
    entrance_per_perimeter_m: float = 100.0  # target entrance count ≈ façade length / this (clipped to [min, max]); real ground floors: one entrance per ~80–120 m of façade
    entrance_max_stub_m: float = 40.0  # an outer-loop node may become an entrance if its stub through the shop band is at most this long
    atrium_per_area_m2: float = 8000.0  # target atrium count ≈ floor area / this (clipped to [1, max_atria])
    max_atria: int = 4
    atrium_spread: float = 0.22  # atria centroids at least this × sqrt(floor area) apart (distribute along the mall, not clustered)
    atrium_area_max: float = 900.0
    min_atrium_area: float = 80.0
    junction_pad: float = 1.25  # pad side = width × this
    smooth_radius: float = 0.0  # m – closing radius for the corridor body (fills notches at bends); 0 = auto: 0.5 × main width
    fill_hole_area: float = 60.0  # m² – holes in the corridor body smaller than this become plaza (kiosk-size islands are noise)
    fillet: bool = True  # round joins and caps (real corridors are smooth ribbons, no mitre spikes)


@dataclass
class CorridorPlan:
    corridors_main: Polygon | MultiPolygon
    corridors_secondary: Polygon | MultiPolygon
    entrances: list[dict]  # {node, point, stub: Polygon}
    atria: list[Polygon]
    edge_class: dict[tuple[str, str], str]
    vertical_cores: list[dict] = field(default_factory=list)  # {node, point, polygon}: dead ends deep inside = stairs / escalators
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


def _smooth(geom, r: float, fill_hole_area: float = 0.0):  # noqa: ANN001, ANN202
    """Morphological *closing* (dilate then erode by ``r``): fills notches and re-entrant spikes narrower than 2r so
    two corridors meeting at an angle read as one continuous ribbon with a rounded inner corner. Holes smaller than
    ``fill_hole_area`` (tiny islands between crowded corridors) are filled – they read as plaza, not as shops.
    No opening step – it would eat thin secondary corridors."""
    if geom is None or geom.is_empty:
        return geom
    g = geom.buffer(r, join_style="round").buffer(-r, join_style="round").buffer(0) if r > 0 else geom
    if fill_hole_area > 0:
        parts = []
        for p in _polys(g):
            keep = [ring for ring in p.interiors if Polygon(ring).area >= fill_hole_area]
            parts.append(Polygon(p.exterior, keep))
        g = unary_union(parts) if parts else g
    return g


def classify_edges(g: nx.Graph, quantile: float, roles: dict[str, str] | None = None, skeleton_nodes: set[str] | None = None) -> dict[tuple[str, str], str]:
    """Corridor hierarchy.

    * With ``skeleton_nodes`` (the Stage-1 prototype = the *main* corridor system by construction): an edge between two
      skeleton nodes is **main**, every edge touching a new node is **secondary**. This is the designer's reading of the
      two-stage pipeline (skeleton = wide primary loop/spine, expansion = secondary corridors).
    * Without a skeleton (e.g. a real complete network): main = outer loop or top ``1-quantile`` edge betweenness."""
    out = {}
    if skeleton_nodes:
        for u, v in g.edges:
            out[(u, v)] = "main" if (u in skeleton_nodes and v in skeleton_nodes) else "secondary"
        if any(c == "main" for c in out.values()):
            return out
    bc = nx.edge_betweenness_centrality(g) if g.number_of_edges() else {}
    vals = np.array(list(bc.values())) if bc else np.array([0.0])
    thr = float(np.quantile(vals, quantile)) if len(vals) else 0.0
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
    # the main corridor is always readably wider than the secondary one: secondary >= min_width, main >= min_width / factor
    w = float(np.clip(w, prm.min_width / prm.secondary_factor, prm.main_width))
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
    stub = LineString([p, fp]).buffer(width / 2, cap_style="flat", join_style="round").buffer(width * 0.02)
    return stub, fp


def _sides(p: Polygon) -> tuple[float, float]:
    xs, ys = np.array(p.minimum_rotated_rectangle.exterior.coords[:-1]).T
    return float(np.hypot(xs[1] - xs[0], ys[1] - ys[0])), float(np.hypot(xs[2] - xs[1], ys[2] - ys[1]))


def _aspect(p: Polygon) -> float:
    e1, e2 = _sides(p)
    return max(e1, e2) / max(min(e1, e2), 1e-9)


def render_corridors(topology: TopologyGraph, positions: dict[str, tuple[float, float]], outline: Outline, roles: dict[str, str] | None = None, params: RenderParams | None = None, skeleton_nodes: set[str] | None = None, entrance_points: list[tuple[float, float]] | None = None) -> CorridorPlan:
    """``skeleton_nodes``: Stage-1 prototype nodes → skeleton–skeleton edges are the main corridors (see
    :func:`classify_edges`). Dead ends (degree-1 nodes) become entrances when the façade is within
    ``prm.entrance_reach`` × main width, otherwise they are marked as vertical circulation (stairs/escalator cores) –
    a real mall corridor never simply stops. ``entrance_points`` (renovation): existing entrance positions that are kept –
    each is connected to the nearest corridor node by a stub and counts towards the entrance target first."""
    prm = params or RenderParams()
    g = to_networkx(topology)
    pos = {k: np.asarray(v, float) for k, v in positions.items()}
    site = outline.polygon
    cls = classify_edges(g, prm.main_quantile, roles, skeleton_nodes)
    main_segs, sec_segs = [], []
    for (u, v), c in cls.items():
        if u in pos and v in pos and np.linalg.norm(pos[u] - pos[v]) > 1e-6:
            (main_segs if c == "main" else sec_segs).append(LineString([pos[u], pos[v]]))
    w_main, w_sec = widths_for(site.area, sum(s.length for s in main_segs), sum(s.length for s in sec_segs), prm)
    join = "round" if prm.fillet else "mitre"
    cap = "round" if prm.fillet else "flat"
    # buffer the *merged* centre-lines (one MultiLineString) so that consecutive segments join smoothly instead of
    # overlapping as separate rectangles (which leaves notches at every bend)
    main = unary_union(main_segs).buffer(w_main / 2, cap_style=cap, join_style=join) if main_segs else Polygon()
    sec = unary_union(sec_segs).buffer(w_sec / 2, cap_style=cap, join_style=join) if sec_segs else Polygon()
    # junction pads: a slightly wider node where >= 3 corridors meet, added to the layer of the widest incident edge
    # (never as a free-standing disc: a pad on secondary-only junctions goes into the secondary layer)
    pads_main, pads_sec = [], []
    for v in g.nodes:
        if g.degree(v) >= 3 and v in pos:
            if any(cls.get((v, u), cls.get((u, v))) == "main" for u in g.neighbors(v)):
                pads_main.append(Point(pos[v]).buffer(w_main * prm.junction_pad / 2))
            else:
                pads_sec.append(Point(pos[v]).buffer(w_sec * prm.junction_pad / 2))
    if pads_main:
        main = unary_union([main, *pads_main])
    if pads_sec:
        sec = unary_union([sec, *pads_sec])
    r_s = prm.smooth_radius if prm.smooth_radius > 0 else 0.5 * w_main
    # smooth the whole corridor body at once so main/secondary join seamlessly, then split by hierarchy
    body = _smooth(unary_union([main, sec]), r_s, prm.fill_hole_area).intersection(site).buffer(0)
    main = _smooth(main, r_s, prm.fill_hole_area).intersection(body).buffer(0)
    sec = body.difference(main).buffer(0)
    # drop crumbs (slivers left by the difference) smaller than one junction pad
    sec = unary_union([q for q in _polys(sec) if q.area >= (w_sec * prm.junction_pad) ** 2 * 0.5]) if _polys(sec) else Polygon()
    main = unary_union([q for q in _polys(main) if q.area >= (w_main * prm.junction_pad) ** 2 * 0.5]) if _polys(main) else Polygon()
    # entrances: every façade-near dead end, then outer-loop nodes spread along the façade until the target count
    entrances: list[dict] = []
    vertical_cores: list[dict] = []
    side = w_main * prm.junction_pad
    n_target = int(np.clip(round(site.exterior.length / prm.entrance_per_perimeter_m), prm.min_entrances, prm.max_entrances))

    def _core(v: str) -> None:
        vertical_cores.append({"node": v, "point": pos[v], "polygon": Point(pos[v]).buffer(side / 2, cap_style="square")})

    def _try_entrance(v: str, hint: np.ndarray | None, kind: str, max_len: float) -> bool:
        stub, fp = _stub_to_facade(pos[v], site, hint, min(prm.entrance_width, w_main))
        if fp is None or np.linalg.norm(fp - pos[v]) > max_len:
            return False
        if any(np.linalg.norm(fp - e["point"]) < prm.entrance_spacing for e in entrances):
            return False
        stub_geom = stub.intersection(site).buffer(0) if stub is not None else Polygon()
        stub_geom = max(_polys(stub_geom), key=lambda q: q.area) if _polys(stub_geom) else Polygon()  # clipping can split the stub
        entrances.append({"node": v, "point": fp, "stub": stub_geom, "kind": kind})
        return True

    # existing entrances (renovation) first: stub from the nearest node that can reach the façade point in a short run
    claimed: set[str] = set()
    for ep in entrance_points or []:
        ep = np.asarray(ep, float)
        fp = np.array(site.exterior.interpolate(site.exterior.project(Point(ep))).coords[0])
        if any(np.linalg.norm(fp - e["point"]) < prm.entrance_spacing * 0.5 for e in entrances):
            continue
        order = sorted((v for v in g.nodes if v in pos and v not in claimed), key=lambda v: (g.degree(v) != 1, np.linalg.norm(pos[v] - fp)))
        for v in order[:3]:
            if np.linalg.norm(pos[v] - fp) > prm.entrance_max_stub_m:
                break
            stub = LineString([pos[v], fp]).buffer(min(prm.entrance_width, w_main) / 2, cap_style="flat").intersection(site).buffer(0)
            stub = max(_polys(stub), key=lambda q: q.area) if _polys(stub) else Polygon()
            entrances.append({"node": v, "point": fp, "stub": stub, "kind": "existing"})
            claimed.add(v)
            break
    dead = sorted((v for v in g.nodes if v in pos and g.degree(v) == 1 and v not in claimed), key=lambda v: site.exterior.distance(Point(pos[v])))
    for v in dead:
        u = next(iter(g.neighbors(v)))
        dist = site.exterior.distance(Point(pos[v]))
        # a dead end deep inside the floor is not an entrance – it is where a stair / escalator core sits
        if dist > prm.entrance_reach * w_main or len(entrances) >= prm.max_entrances or not _try_entrance(v, pos[v] - pos[u], "dead_end", np.inf):
            _core(v)
    # outer-loop nodes: greedy farthest-point selection on the façade so entrances are distributed around the building
    loop = [v for v in g.nodes if v in pos and g.degree(v) >= 2 and roles and roles.get(v) == "outer"]
    if not loop and roles is None:
        loop = [v for v in g.nodes if v in pos and g.degree(v) >= 2 and site.exterior.distance(Point(pos[v])) < prm.entrance_max_stub_m]
    fps = {v: np.array(site.exterior.interpolate(site.exterior.project(Point(pos[v]))).coords[0]) for v in loop}
    loop = [v for v in loop if np.linalg.norm(fps[v] - pos[v]) <= prm.entrance_max_stub_m]
    while loop and len(entrances) < n_target:
        if entrances:
            v = max(loop, key=lambda q: min(np.linalg.norm(fps[q] - e["point"]) for e in entrances))
        else:  # first one: the node closest to the façade
            v = min(loop, key=lambda q: np.linalg.norm(fps[q] - pos[q]))
        loop.remove(v)
        if min((np.linalg.norm(fps[v] - e["point"]) for e in entrances), default=np.inf) < prm.entrance_spacing:
            break  # the best remaining candidate is already too close: the façade is saturated
        _try_entrance(v, None, "loop", prm.entrance_max_stub_m)
    # atria: voids enclosed by corridors (faces of the whole centre-line network), preferring compact faces that open
    # onto a main corridor, distributed along the mall (≈ one per ``atrium_per_area_m2``) rather than clustered
    atria: list[Polygon] = []
    corr_body = unary_union([main, sec])
    cands_a: list[tuple[float, Polygon]] = []
    if main_segs or sec_segs:
        for f in polygonize(unary_union([*main_segs, *sec_segs])):
            hole = f.buffer(-w_sec / 2 - r_s * 0.5, join_style="round").buffer(0)
            hole = hole.difference(corr_body.buffer(r_s * 0.3)).buffer(0)  # never overlap a corridor
            hole = max(_polys(hole), key=lambda q: q.area) if _polys(hole) else Polygon()
            if hole.is_empty or hole.area < prm.min_atrium_area:
                continue
            if hole.area > prm.atrium_area_max:  # a large block: the atrium is its compact centre, ringed by island shops
                lo, hi = 0.0, float(np.sqrt(hole.area))
                for _ in range(25):
                    mid = (lo + hi) / 2
                    h2 = hole.buffer(-mid)
                    if h2.is_empty or h2.area < prm.atrium_area_max:
                        hi = mid
                    else:
                        lo = mid
                h2 = hole.buffer(-lo)
                h2 = max(_polys(h2), key=lambda q: q.area) if _polys(h2) else Polygon()
                if not h2.is_empty and _aspect(h2) > 4.0:  # long sliver between parallel corridors: take a square at its centre
                    c = h2.centroid
                    h2 = hole.intersection(c.buffer(float(np.sqrt(prm.atrium_area_max)) / 2, cap_style="square")).buffer(0)
                    h2 = max(_polys(h2), key=lambda q: q.area) if _polys(h2) else Polygon()
                hole = h2
            if hole.is_empty or hole.area < prm.min_atrium_area:
                continue
            e1, e2 = _sides(hole)
            if min(e1, e2) < 1.5 * w_main or max(e1, e2) / max(min(e1, e2), 1e-9) > 4.0:
                continue
            compact = hole.area / max(hole.minimum_rotated_rectangle.area, 1e-9)
            on_main = 2.0 if (not main.is_empty and hole.buffer(w_main).intersects(main)) else 1.0
            cands_a.append((on_main * compact * min(hole.area, prm.atrium_area_max) / prm.atrium_area_max, hole))
    n_atr = int(np.clip(round(site.area / prm.atrium_per_area_m2), 1, prm.max_atria))
    spread = prm.atrium_spread * float(np.sqrt(site.area))
    for _, hole in sorted(cands_a, key=lambda t: -t[0]):
        if len(atria) >= n_atr:
            break
        if all(hole.centroid.distance(a.centroid) >= spread for a in atria):
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
    for i, vc in enumerate(vertical_cores):
        units.append(SpaceUnit(unit_id=f"VC{i}", kind="vertical_core", centroid=(float(vc["point"][0]), float(vc["point"][1])), attached_to=[vc["node"]], polygon=[tuple(map(float, c)) for c in vc["polygon"].exterior.coords[:-1]], area=float(vc["polygon"].area)))
    corr = unary_union([main, sec, *[e["stub"] for e in entrances if not e["stub"].is_empty]])
    diag = {
        "main_width_m": w_main, "secondary_width_m": w_sec,
        "n_main_edges": sum(1 for c in cls.values() if c == "main"), "n_secondary_edges": sum(1 for c in cls.values() if c == "secondary"),
        "corridor_area_m2": float(corr.area), "corridor_ratio": float(corr.area / max(site.area, 1e-9)), "n_entrances": len(entrances), "n_atria": len(atria), "n_vertical_cores": len(vertical_cores), "n_dead_ends": sum(1 for v in g.nodes if g.degree(v) == 1),
        "atrium_area_m2": float(sum(a.area for a in atria)), "main_length_m": float(sum(s.length for s in main_segs)), "secondary_length_m": float(sum(s.length for s in sec_segs)),
    }
    return CorridorPlan(corridors_main=main, corridors_secondary=sec, entrances=entrances, atria=atria, edge_class=cls, vertical_cores=vertical_cores, units=units, diagnostics=diag)


__all__ = ["CorridorPlan", "RenderParams", "classify_edges", "render_corridors"]
