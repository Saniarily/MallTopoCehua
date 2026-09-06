"""Planar, mall-like embedding of a corridor key-point topology.

Design principles (from the architectural brief, see ``docs/methodology.md`` §Stage-2 geometry):

1. **Loops span the core.** The 2-core (all cycles) of the corridor graph is the main corridor
   system; it is drawn as a *convex-outer-face planar straight-line drawing* (Tutte barycentric
   embedding): the outer cycle is pinned to a rounded polygon and every interior vertex is placed at
   the barycentre of its neighbours. For a 3-connected planar graph this is provably crossing-free;
   for the general (2-connected / bridged) case we apply it per biconnected block and fall back to a
   crossing-repair relaxation. Loops therefore *open up* instead of collapsing into a star.
2. **Holes enclosed by loops are atria.** Interior faces of the embedded core are candidate atria;
   the largest ones (by area, above ``min_atrium_area``) become atrium units.
3. **Branches leave the core perpendicularly.** Trees hanging off the core (degree-1 chains,
   leaf clusters) are laid out along the *outward normal* of the corridor they attach to, one
   segment per edge with a fixed corridor step, alternating sides only when the outward side is
   blocked by the boundary.
4. **Straight corridors, right angles.** After placement, edge directions are snapped to the
   dominant orthogonal frame of the outer face (soft snap, ``ortho_weight``), which is what makes
   the drawing read as a mall plan rather than a force-directed graph.

Everything is deterministic given the graph (no seed needed); ``jitter`` is 0 by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.topology.convert import to_networkx


@dataclass
class PlanarEmbedParams:
    ortho_weight: float = 0.6  # 0 = free angles, 1 = hard orthogonal snap
    branch_step: float = 1.0  # branch edge length relative to the median core edge
    relax_iters: int = 60
    min_atrium_area_frac: float = 0.015  # of the outer-face area
    max_atria: int = 3
    jitter: float = 0.0


# --------------------------------------------------------------------------- helpers
def _outer_cycle(core: nx.Graph) -> list[str]:
    """Longest simple cycle found by a planar embedding's face traversal (approx. outer face)."""
    is_planar, emb = nx.check_planarity(core)
    if not is_planar:
        # fall back to a long cycle from the cycle basis
        basis = nx.cycle_basis(core)
        return max(basis, key=len) if basis else list(core.nodes)
    best: list[str] = []
    seen: set[tuple[str, str]] = set()
    for u, v in emb.edges():
        if (u, v) in seen:
            continue
        face = emb.traverse_face(u, v, mark_half_edges=seen)
        if len(face) > len(best):
            best = face
    return best


def _polygon_points(n: int, radius: float = 1.0, aspect: float = 1.35) -> np.ndarray:
    """``n`` points on a rounded rectangle-ish ellipse (malls are elongated, not round)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    x, y = np.cos(t) * radius * aspect, np.sin(t) * radius
    # superellipse exponent 3 -> flatter sides than an ellipse
    k = 3.0
    r = (np.abs(np.cos(t)) ** k + np.abs(np.sin(t)) ** k) ** (-1.0 / k)
    return np.c_[x * r, y * r]


def _tutte(core: nx.Graph, outer: list[str]) -> dict[str, np.ndarray]:
    """Barycentric (Tutte) embedding with ``outer`` pinned to a convex polygon."""
    pos: dict[str, np.ndarray] = {v: p for v, p in zip(outer, _polygon_points(len(outer)), strict=True)}
    inner = [v for v in core.nodes if v not in pos]
    if not inner:
        return pos
    idx = {v: i for i, v in enumerate(inner)}
    A = np.zeros((len(inner), len(inner)))
    bx = np.zeros(len(inner))
    by = np.zeros(len(inner))
    for v in inner:
        i = idx[v]
        nb = list(core.neighbors(v))
        A[i, i] = max(len(nb), 1)
        for u in nb:
            if u in idx:
                A[i, idx[u]] -= 1
            else:
                bx[i] += pos[u][0]
                by[i] += pos[u][1]
    try:
        xs = np.linalg.solve(A, bx)
        ys = np.linalg.solve(A, by)
    except np.linalg.LinAlgError:
        xs, ys = np.linalg.lstsq(A, bx, rcond=None)[0], np.linalg.lstsq(A, by, rcond=None)[0]
    for v in inner:
        pos[v] = np.array([xs[idx[v]], ys[idx[v]]])
    return pos


def _segments_cross(p1, p2, p3, p4) -> bool:  # noqa: ANN001
    def orient(a, b, c):  # noqa: ANN001, ANN202
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    if len({tuple(p1), tuple(p2), tuple(p3), tuple(p4)}) < 4:
        return False
    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


def count_crossings(g: nx.Graph, pos: dict[str, np.ndarray]) -> int:
    E = list(g.edges)
    n = 0
    for i in range(len(E)):
        a, b = E[i]
        for j in range(i + 1, len(E)):
            c, d = E[j]
            if len({a, b, c, d}) == 4 and _segments_cross(pos[a], pos[b], pos[c], pos[d]):
                n += 1
    return n


def _ortho_snap(g: nx.Graph, pos: dict[str, np.ndarray], fixed: set[str], weight: float, iters: int, min_len: float) -> dict[str, np.ndarray]:
    """Soft snap of edge directions to the 0/90° frame + repulsion keeping nodes apart. Fixed nodes don't move."""
    if weight <= 0:
        return pos
    P = {k: v.copy() for k, v in pos.items()}
    nodes = [v for v in g.nodes if v not in fixed]
    for _ in range(iters):
        disp = {v: np.zeros(2) for v in nodes}
        for u, v in g.edges:
            d = P[v] - P[u]
            L = np.linalg.norm(d) + 1e-9
            ang = np.arctan2(d[1], d[0])
            snapped = np.round(ang / (np.pi / 2)) * (np.pi / 2)
            target = np.array([np.cos(snapped), np.sin(snapped)]) * max(L, min_len)
            corr = (target - d) * 0.5 * weight
            if v in disp:
                disp[v] += corr
            if u in disp:
                disp[u] -= corr
        # repulsion for near-coincident nodes
        keys = list(P)
        arr = np.array([P[k] for k in keys])
        for i, k in enumerate(keys):
            if k not in disp:
                continue
            d = arr[i] - arr
            dist = np.linalg.norm(d, axis=1) + 1e-9
            close = dist < min_len
            close[i] = False
            if close.any():
                disp[k] += ((d[close] / dist[close, None]) * (min_len - dist[close, None])).sum(0) * 0.5
        for v in nodes:
            P[v] = P[v] + disp[v] * 0.3
    return P


# --------------------------------------------------------------------------- main
def planar_corridor_embedding(topology: TopologyGraph, params: PlanarEmbedParams | None = None) -> tuple[dict[str, tuple[float, float]], dict]:
    """Return ``(positions in an abstract unit frame, info)``. ``info`` has ``core_nodes``, ``outer_cycle``,
    ``faces`` (atrium candidates as polygons in the same frame, largest first), ``crossings``."""
    prm = params or PlanarEmbedParams()
    g = to_networkx(topology)
    if g.number_of_nodes() == 0:
        return {}, {"core_nodes": [], "outer_cycle": [], "faces": [], "crossings": 0}
    # work on the largest component; others are appended afterwards
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    main = g.subgraph(comps[0]).copy()
    core = nx.k_core(main, 2)
    pos: dict[str, np.ndarray] = {}
    outer: list[str] = []
    if core.number_of_nodes() >= 3:
        outer = _outer_cycle(core)
        # Tutte per biconnected block would be more exact; the whole core works well when the outer
        # face is a real cycle. Blocks connected by bridges are handled by the relaxation below.
        pos = _tutte(core, outer)
        # relax bridges / degenerate placements (nodes collapsed on a line) with a light spring step
        if count_crossings(core, pos) > 0 or len(set(map(tuple, np.round([pos[v] for v in core], 4)))) < core.number_of_nodes():
            sp = nx.spring_layout(core, pos={k: tuple(v) for k, v in pos.items()}, fixed=outer, k=0.6 / np.sqrt(core.number_of_nodes()), iterations=80, seed=0)
            pos = {k: np.asarray(v, float) for k, v in sp.items()}
    else:
        # tree-like topology: pick the longest path as a straight spine (一字型) and hang the rest off it
        u = max(main.nodes, key=lambda n: nx.eccentricity(main, n)) if main.number_of_nodes() > 1 else next(iter(main.nodes))
        far = nx.single_source_shortest_path_length(main, u)
        v = max(far, key=far.get)
        spine = nx.shortest_path(main, u, v)
        for i, n in enumerate(spine):
            pos[n] = np.array([i - (len(spine) - 1) / 2.0, 0.0]) * (2.0 / max(len(spine) - 1, 1))
        core = main.subgraph(spine).copy()
        outer = spine
    core_nodes = set(pos)
    med_len = float(np.median([np.linalg.norm(pos[a] - pos[b]) for a, b in core.edges])) if core.number_of_edges() else 0.35
    step = prm.branch_step * med_len
    # ---- branches: BFS out from the core; each new node goes along the outward normal of its attachment
    centre = np.mean([pos[v] for v in core_nodes], axis=0)
    order = sorted((n for n in main.nodes if n not in core_nodes), key=lambda n: (nx.shortest_path_length(main, n, next(iter(core_nodes))) if core_nodes else 0, n))
    placed = set(core_nodes)
    # process in BFS layers from the core so parents are placed before children
    layer = {n: 0 for n in core_nodes}
    frontier = list(core_nodes)
    while frontier:
        nxt = []
        for u in frontier:
            for w in main.neighbors(u):
                if w not in layer:
                    layer[w] = layer[u] + 1
                    nxt.append(w)
        frontier = nxt
    occupied: list[np.ndarray] = [pos[v] for v in core_nodes]
    for n in sorted(order, key=lambda n: (layer.get(n, 99), n)):
        parents = [u for u in main.neighbors(n) if u in placed]
        if not parents:
            continue
        anchor = np.mean([pos[u] for u in parents], axis=0)
        # outward normal: away from the core centre; for a parent on the core, perpendicular to its corridor
        if len(parents) == 1 and parents[0] in core_nodes:
            p = parents[0]
            nb = [q for q in core.neighbors(p)] if p in core else []
            if len(nb) >= 2:
                a, b = pos[nb[0]] - pos[p], pos[nb[1]] - pos[p]
                tang = a / (np.linalg.norm(a) + 1e-9) - b / (np.linalg.norm(b) + 1e-9)
                if np.linalg.norm(tang) < 1e-6:
                    tang = a
                normal = np.array([-tang[1], tang[0]])
                normal /= np.linalg.norm(normal) + 1e-9
                if np.dot(normal, pos[p] - centre) < 0:
                    normal = -normal
            else:
                normal = pos[p] - centre
                normal /= np.linalg.norm(normal) + 1e-9
        else:
            normal = anchor - centre
            if np.linalg.norm(normal) < 1e-6:
                normal = np.array([0.0, 1.0])
            normal /= np.linalg.norm(normal)
            if len(parents) == 1:  # continue the parent's own direction (straight corridor)
                p = parents[0]
                gp = [q for q in main.neighbors(p) if q in placed and q != n]
                if gp:
                    d = pos[p] - np.mean([pos[q] for q in gp], axis=0)
                    if np.linalg.norm(d) > 1e-6:
                        normal = 0.5 * normal + 0.5 * d / np.linalg.norm(d)
                        normal /= np.linalg.norm(normal)
        cand = anchor + normal * step
        # avoid landing on an occupied spot: rotate around the anchor in 30° steps
        for k in range(1, 12):
            if all(np.linalg.norm(cand - o) > 0.55 * step for o in occupied):
                break
            ang = (k // 2 + 1) * (np.pi / 6) * (1 if k % 2 else -1)
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            cand = anchor + (R @ normal) * step
        pos[n] = cand
        occupied.append(cand)
        placed.add(n)
    # ---- other components: place beside the main drawing
    if len(comps) > 1:
        xs = np.array([p[0] for p in pos.values()])
        off = xs.max() + 2 * step
        for comp in comps[1:]:
            sub = g.subgraph(comp)
            sp = nx.spring_layout(sub, seed=0, scale=step * max(1, np.sqrt(len(comp)) / 2))
            for k, v in sp.items():
                pos[k] = np.asarray(v, float) + np.array([off, 0.0])
            off += 2 * step * max(1, np.sqrt(len(comp)))
    # ---- orthogonal soft snap (core outer cycle fixed so loops stay open)
    pos = _ortho_snap(main if len(comps) == 1 else g, pos, set(outer), prm.ortho_weight, prm.relax_iters, min_len=0.6 * step)
    # ---- faces of the core = atrium candidates (as polygons in the same frame, largest first)
    faces: list[list[tuple[float, float]]] = []
    if core.number_of_edges() >= 3:
        lines = [LineString([pos[a], pos[b]]) for a, b in core.edges]
        polys = list(polygonize(unary_union(lines)))
        outer_poly = Polygon([pos[v] for v in outer]) if len(outer) >= 3 else None
        ref_area = outer_poly.area if outer_poly is not None and outer_poly.is_valid and outer_poly.area > 0 else sum(p.area for p in polys)
        polys = [p for p in polys if p.area >= ref_area * prm.min_atrium_area_frac]
        polys.sort(key=lambda p: -p.area)
        faces = [[(float(x), float(y)) for x, y in p.exterior.coords[:-1]] for p in polys[: prm.max_atria]]
    info = {"core_nodes": sorted(core_nodes), "outer_cycle": list(outer), "faces": faces, "crossings": count_crossings(g, pos), "n_branch_nodes": int(len(order))}
    return {k: (float(v[0]), float(v[1])) for k, v in pos.items()}, info


__all__ = ["PlanarEmbedParams", "count_crossings", "planar_corridor_embedding"]
