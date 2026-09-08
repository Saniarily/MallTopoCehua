"""Fit an M key-point network into an outline (Stage 3 core).

Pipeline (deterministic given ``seed``; ``n_restarts`` candidates are scored and the best returned):

1. **Abstract planar drawing** of the topology (``planar_corridor_embedding`` from Stage 2: loops opened
   by a Tutte embedding, branches perpendicular).
2. **Similarity alignment** to the *inset region* (outline shrunk by one shop depth): scale to the inset
   bbox, try the 8 axis-aligned rotations/reflections × PCA-frame rotation, keep the pose with the best
   (inside ratio, medial-axis proximity) score.
3. **Constrained relaxation** (gradient-style iterations):
   * outer-loop nodes → attracted to the inset boundary (corridors run one shop depth inside the façade);
   * tree/branch nodes → attracted to the medial axis of the inset region (wing centre-lines);
   * edges → soft snap to the outline's dominant wall directions (straight corridors, right angles);
   * repulsion between non-adjacent nodes (≥ ``min_spacing``) and edge–node clearance;
   * every step is projected back inside the inset region; moves that create a crossing are rejected.
4. **Corner snapping**: outer-loop nodes within ``snap_dist`` of an inset-boundary corner are snapped to it
   (corridors turn where the façade turns).

The result carries the positions (metres, outline frame), the node roles (outer loop / inner core /
branch / leaf) and diagnostics. Geometry only – topology is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from mall_space_planner.geometry.planar_embed import PlanarEmbedParams, count_crossings, crossing_matrix, planar_corridor_embedding
from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.stage3.outline import Outline
from mall_space_planner.stage3.skeleton import dominant_directions, inset_region, medial_axis_graph, nearest_on_lines
from mall_space_planner.topology.convert import to_networkx


@dataclass
class FitParams:
    shop_depth: float = 14.0  # m – outer corridor sits this far inside the façade (capped by ``depth_frac`` × mean floor width)
    depth_frac: float = 0.28  # mean floor width = 2·area/perimeter; depth ≤ depth_frac × that
    depth_area_coef: float = 0.18  # real floors (195 test floors): outer M nodes sit ≈ 0.18·√area inside the façade (median; IQR 0.155–0.215)
    min_angle_deg: float = 60.0  # corridors meeting at a node open at least this much (real: only 13% of node angles < 60°)
    w_angle: float = 0.8  # weight of the angle-opening force
    n_offsets: int = 12  # rotational offsets tried when pinning the outer loop onto the inset boundary
    min_spacing: float = 12.0  # m – minimum distance between non-adjacent key points
    iters: int = 120
    step: float = 0.25
    w_boundary: float = 1.0  # outer loop -> inset boundary
    w_axis: float = 0.8  # branches -> medial axis
    w_ortho: float = 0.6  # edge direction -> wall frame
    w_repel: float = 2.0
    w_straight: float = 0.4  # degree-2 nodes: keep the two edges collinear
    snap_dist: float = 6.0  # corner snapping distance
    n_restarts: int = 6
    raster_px: float = 1.0
    w_leaf_facade: float = 0.6  # dead-end key points are entrances: pull (non-anchored) leaves towards the inset boundary
    anchor_jitter: float = 0.35  # renovation mode: restarts jitter the *new* nodes by this × min_spacing (anchored nodes never move)


@dataclass
class FitResult:
    positions: dict[str, tuple[float, float]]
    roles: dict[str, str]  # outer | core | branch | leaf
    inset: Polygon
    axis_lines: list[LineString]
    frame_angles: list[float]
    score: float
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------- utilities
def _roles(g: nx.Graph, info: dict) -> dict[str, str]:
    core = set(info.get("core_nodes", []))
    outer = set(info.get("outer_cycle", []))
    r = {}
    for v in g.nodes:
        if v in outer:
            r[v] = "outer"
        elif v in core:
            r[v] = "core"
        elif g.degree(v) == 1:
            r[v] = "leaf"
        else:
            r[v] = "branch"
    return r


def effective_depth(outline: Outline, shop_depth: float, depth_frac: float, depth_area_coef: float = 0.0) -> float:
    """Inset depth of the outer corridor. ``shop_depth`` is the nominal value; with ``depth_area_coef`` > 0 the depth
    scales with the floor (``coef·√area``, calibrated on real floors) and ``shop_depth`` acts as a floor; both are
    capped by ``depth_frac`` × mean floor width so narrow wings keep a usable inset."""
    poly = outline.polygon
    mean_w = 2.0 * poly.area / max(poly.exterior.length, 1e-9)
    d = shop_depth if depth_area_coef <= 0 else max(shop_depth, depth_area_coef * float(np.sqrt(poly.area)))
    return float(max(3.0, min(d, depth_frac * mean_w)))


def _arc_positions(ring: np.ndarray, n: int, offset: float, reverse: bool) -> np.ndarray:
    """``n`` points along a closed ring (Nx2, not repeated) at equal arc length, starting at fraction ``offset``."""
    R = ring[::-1] if reverse else ring
    seg = np.linalg.norm(np.diff(np.vstack([R, R[:1]]), axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    s = (offset * L + np.arange(n) * L / n) % L
    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(R) - 1)
    frac = (s - cum[idx]) / np.maximum(seg[idx], 1e-9)
    nxt = (idx + 1) % len(R)
    return R[idx] + (R[nxt] - R[idx]) * frac[:, None]


def _corner_aware_arc(ring: np.ndarray, n: int, offset: float, reverse: bool, snap: float) -> np.ndarray:
    """Equal-arc distribution, then pull points that are within ``snap`` of a ring vertex onto it (corridors turn
    where the façade turns); a vertex is used at most once."""
    P = _arc_positions(ring, n, offset, reverse)
    used = set()
    for i in range(len(P)):
        d = np.linalg.norm(ring - P[i], axis=1)
        j = int(d.argmin())
        if d[j] < snap and j not in used:
            P[i] = ring[j]
            used.add(j)
    return P


def _tutte_interior(g: nx.Graph, fixed: dict[str, np.ndarray], inner: list[str]) -> dict[str, np.ndarray]:
    """Barycentric placement of ``inner`` nodes with ``fixed`` positions given (Laplacian solve)."""
    if not inner:
        return {}
    idx = {v: i for i, v in enumerate(inner)}
    A = np.zeros((len(inner), len(inner)))
    b = np.zeros((len(inner), 2))
    for v in inner:
        i = idx[v]
        nb = [u for u in g.neighbors(v) if u in idx or u in fixed]
        A[i, i] = max(len(nb), 1)
        for u in nb:
            if u in idx:
                A[i, idx[u]] -= 1
            else:
                b[i] += fixed[u]
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, b, rcond=None)[0]
    return {v: X[idx[v]] for v in inner}


def _init_on_inset(g: nx.Graph, roles: dict[str, str], outer_cycle: list[str], inset: Polygon, lines: list[LineString], depth: float, snap: float, n_offsets: int) -> list[dict[str, np.ndarray]]:
    """Candidate initial placements: outer loop pinned onto the inset boundary (several offsets / both
    directions), interior core by Tutte, branches walked outward from their attachment towards the medial axis."""
    ring = np.array(inset.exterior.coords[:-1])
    hull = inset.convex_hull
    core_inner = [v for v, r in roles.items() if r == "core"]
    branch = [v for v, r in roles.items() if r in ("branch", "leaf")]
    cands: list[dict[str, np.ndarray]] = []
    if len(outer_cycle) >= 3:
        for k in range(n_offsets):
            for rev in (False, True):
                P = _corner_aware_arc(ring, len(outer_cycle), k / n_offsets, rev, snap)
                pos = {v: P[i] for i, v in enumerate(outer_cycle)}
                pos.update(_tutte_interior(g, pos, core_inner))
                for v in core_inner:
                    pos[v] = _project_inside(pos[v], inset)
                cands.append(pos)
    else:  # tree-like: longest medial-axis path as spine
        pos: dict[str, np.ndarray] = {}
        if lines:
            spine = max(lines, key=lambda l: l.length)
            # longest simple path in g
            u = max(g.nodes, key=lambda n: nx.eccentricity(g, n)) if g.number_of_nodes() > 1 else next(iter(g.nodes))
            far = nx.single_source_shortest_path_length(g, u)
            v = max(far, key=far.get)
            path = nx.shortest_path(g, u, v)
            for i, n in enumerate(path):
                q = spine.interpolate((i + 0.5) / len(path), normalized=True)
                pos[n] = np.array([q.x, q.y])
            for rev in (False, True):
                pp = dict(pos)
                if rev:
                    ps = [pp[n] for n in path][::-1]
                    for n, q in zip(path, ps, strict=True):
                        pp[n] = q
                cands.append(pp)
        else:
            c = inset.centroid
            cands.append({v: np.array([c.x, c.y]) for v in g.nodes})
    # branches: BFS from placed nodes; each new node = parent + step towards (medial axis point away from centroid)
    cen = np.array(inset.centroid.coords[0])
    step = max(depth, 0.35 * float(np.sqrt(inset.area / max(g.number_of_nodes(), 1))))
    for pos in cands:
        placed = set(pos)
        frontier = [v for v in g.nodes if v in placed]
        while frontier:
            nxt = []
            for u in frontier:
                for w in g.neighbors(u):
                    if w in placed:
                        continue
                    par = [q for q in g.neighbors(w) if q in placed]
                    base = np.mean([pos[q] for q in par], axis=0)
                    d = base - cen
                    d = d / (np.linalg.norm(d) + 1e-9)
                    cand = base + d * step
                    if lines:
                        cand = np.array(nearest_on_lines(lines, tuple(cand)))
                        if np.linalg.norm(cand - base) < 0.3 * step:  # axis point too close: keep walking outward
                            cand = base + d * step
                    pos[w] = _project_inside(cand, inset)
                    placed.add(w)
                    nxt.append(w)
            frontier = nxt
        for v in g.nodes:  # disconnected leftovers
            if v not in pos:
                pos[v] = np.array(inset.representative_point().coords[0])
    return cands


def _init_anchored(g: nx.Graph, fixed: dict[str, np.ndarray], inset: Polygon, lines: list[LineString], depth: float, rng: np.random.RandomState, n: int = 4, jitter: float = 0.0) -> list[dict[str, np.ndarray]]:
    """Renovation start: anchored nodes at their real positions; new nodes with ≥ 2 placed neighbours by Tutte
    (barycentre of the anchored frame), the rest walked outward from their attachment (BFS, towards the medial axis).
    ``n`` candidates differ by a jitter of the new nodes (the first one is unjittered)."""
    new_nodes = [v for v in g.nodes if v not in fixed]
    cands: list[dict[str, np.ndarray]] = []
    cen = np.array(inset.centroid.coords[0])
    step = max(depth, 0.35 * float(np.sqrt(inset.area / max(g.number_of_nodes(), 1))))
    for k in range(max(n, 1)):
        pos = {v: q.copy() for v, q in fixed.items()}
        # interior new nodes (degree ≥ 2, in a new-node cluster that touches an anchored node) → Tutte, solved jointly;
        # dead ends and clusters without any anchored neighbour are placed by the BFS walk below
        sub = g.subgraph(new_nodes)
        inner: list[str] = []
        for comp in nx.connected_components(sub):
            if any(u in fixed for v in comp for u in g.neighbors(v)):
                inner.extend(v for v in comp if g.degree(v) >= 2)
        if inner:
            try:
                sol = _tutte_interior(g, pos, inner)
                for v in inner:
                    if v in sol and np.all(np.isfinite(sol[v])):
                        pos[v] = _project_inside(sol[v], inset)
            except Exception:  # noqa: BLE001  (singular system when a new cluster has no anchored neighbour)
                pass
        # BFS for whatever is still unplaced (chains / leaves hanging off one node)
        placed = set(pos)
        frontier = [v for v in g.nodes if v in placed]
        while frontier:
            nxt = []
            for u in frontier:
                for w in g.neighbors(u):
                    if w in placed:
                        continue
                    par = [q for q in g.neighbors(w) if q in placed]
                    base = np.mean([pos[q] for q in par], axis=0)
                    d = base - cen
                    d = d / (np.linalg.norm(d) + 1e-9)
                    cand = base + d * step
                    if g.degree(w) == 1:  # a new dead end is an entrance: head for the inset boundary
                        cand = _boundary_point(inset, base)
                        if np.linalg.norm(cand - base) < 0.3 * step:
                            cand = base + d * step
                    elif lines:
                        cand = np.array(nearest_on_lines(lines, tuple(cand)))
                        if np.linalg.norm(cand - base) < 0.3 * step:
                            cand = base + d * step
                    pos[w] = _project_inside(cand, inset)
                    placed.add(w)
                    nxt.append(w)
            frontier = nxt
        for v in g.nodes:
            if v not in pos:
                pos[v] = np.array(inset.representative_point().coords[0])
        if k > 0 and jitter > 0:
            for v in new_nodes:
                pos[v] = _project_inside(pos[v] + rng.normal(0, jitter, 2), inset)
        cands.append(pos)
    return cands


def _project_inside(p: np.ndarray, poly: Polygon) -> np.ndarray:
    if not np.all(np.isfinite(p)):
        c = poly.representative_point()
        return np.array([c.x, c.y])
    from shapely import contains_xy

    if contains_xy(poly, float(p[0]), float(p[1])):
        return p
    pt = Point(p)
    q = poly.exterior.interpolate(poly.exterior.project(pt))
    # nudge slightly inside
    c = poly.representative_point()
    d = np.array([c.x - q.x, c.y - q.y])
    n = np.linalg.norm(d)
    return np.array([q.x, q.y]) + (d / n * 0.3 if n > 1e-9 else 0)


def _boundary_point(poly: Polygon, p: np.ndarray) -> np.ndarray:
    q = poly.exterior.interpolate(poly.exterior.project(Point(p)))
    return np.array([q.x, q.y])


def _snap_angle(d: np.ndarray, frame: list[float]) -> np.ndarray:
    """Rotate ``d`` to the closest frame direction (frame angles and their perpendiculars)."""
    L = np.linalg.norm(d)
    if L < 1e-9:
        return d
    ang = np.arctan2(d[1], d[0])
    cands = []
    for a in frame:
        for k in range(4):
            cands.append(a + k * np.pi / 2)
    best = min(cands, key=lambda a: abs(((ang - a) + np.pi) % (2 * np.pi) - np.pi))
    return np.array([np.cos(best), np.sin(best)]) * L


def _crossings(g: nx.Graph, pos: dict[str, np.ndarray]) -> int:
    return count_crossings(g, pos)


def node_angles(g: nx.Graph, pos: dict[str, np.ndarray]) -> np.ndarray:
    """All angles (degrees) between pairs of edges meeting at a node."""
    out = []
    for v in g.nodes:
        nb = list(g.neighbors(v))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a, b = np.asarray(pos[nb[i]]) - pos[v], np.asarray(pos[nb[j]]) - pos[v]
                La, Lb = np.linalg.norm(a), np.linalg.norm(b)
                if La < 1e-9 or Lb < 1e-9:
                    continue
                out.append(np.degrees(np.arccos(np.clip(np.dot(a, b) / (La * Lb), -1, 1))))
    return np.asarray(out, float)


def sharp_angle_rate(g: nx.Graph, pos: dict[str, np.ndarray], min_deg: float) -> float:
    A = node_angles(g, pos)
    return float(np.mean(A < min_deg)) if len(A) else 0.0


# ----------------------------------------------------------------------------------------- fitter
class CorridorFitter:
    def __init__(self, params: FitParams | None = None) -> None:
        self.p = params or FitParams()

    # ---- scoring ------------------------------------------------------------------------------
    def score(self, g: nx.Graph, pos: dict[str, np.ndarray], roles: dict[str, str], inset: Polygon, lines: list[LineString], frame: list[float], fixed: set[str] | None = None) -> tuple[float, dict]:
        P = np.array([pos[v] for v in g.nodes])
        movable = [pos[v] for v in g.nodes if not fixed or v not in fixed]
        inside = float(np.mean([inset.contains(Point(p)) or inset.touches(Point(p)) for p in movable])) if movable else 1.0
        cross = _crossings(g, pos)
        # boundary proximity of outer nodes / axis proximity of branch nodes (normalised by shop depth)
        d_b = [inset.exterior.distance(Point(pos[v])) for v in g.nodes if roles[v] == "outer"]
        d_a = [Point(pos[v]).distance(LineString(nearest_on_lines(lines, tuple(pos[v])) and [tuple(pos[v]), nearest_on_lines(lines, tuple(pos[v]))])) if lines else 0.0 for v in g.nodes if roles[v] in ("branch", "leaf")]
        near_b = float(np.mean(d_b)) / self.p.shop_depth if d_b else 0.0
        near_a = float(np.mean(d_a)) / self.p.shop_depth if d_a else 0.0
        # orthogonality: mean angular deviation of edges from the frame (0..45°)
        devs = []
        for u, v in g.edges:
            d = pos[v] - pos[u]
            if np.linalg.norm(d) < 1e-9:
                continue
            ang = np.arctan2(d[1], d[0])
            devs.append(min(abs(((ang - a - k * np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2) for a in frame for k in range(2)))
        ortho = float(np.degrees(np.mean(devs))) / 45.0 if devs else 0.0
        # spacing violations
        keys = list(g.nodes)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P)) * 1e9
        adj = nx.to_numpy_array(g, nodelist=keys) > 0
        # share of *nodes* that have a non-adjacent node closer than min_spacing (pair-fraction was ~0 for any layout)
        viol = float(np.mean(((D < self.p.min_spacing) & ~adj).any(1))) if len(P) > 1 else 0.0
        # coverage: share of the inset within one shop depth of a corridor, estimated on a coarse grid of sample
        # points (a shapely buffer of the whole network costs ~0.3 s per call, this is ~1 ms)
        cover = self._coverage(g, pos, inset)
        # sharp wedges: share of node angles below min_angle (real floors: ≈ 0.13 below 60°)
        sharp = sharp_angle_rate(g, pos, self.p.min_angle_deg)
        s = 3.0 * (1 - inside) + 2.0 * cross + 0.8 * near_b + 0.5 * near_a + 0.8 * ortho + 2.0 * viol + 1.0 * (1 - cover) + 1.5 * sharp
        return s, {"inside_ratio": inside, "crossings": cross, "outer_to_boundary_m": near_b * self.p.shop_depth, "branch_to_axis_m": near_a * self.p.shop_depth, "ortho_deviation_deg": ortho * 45.0, "spacing_violation_rate": viol, "served_area_ratio": cover, "sharp_angle_rate": sharp}

    def _coverage(self, g: nx.Graph, pos: dict[str, np.ndarray], inset: Polygon, n_grid: int = 24) -> float:
        key = id(inset)
        if getattr(self, "_grid_key", None) != key:
            minx, miny, maxx, maxy = inset.bounds
            xs = np.linspace(minx, maxx, n_grid)
            ys = np.linspace(miny, maxy, n_grid)
            G = np.array([(x, y) for x in xs for y in ys])
            from shapely import contains_xy

            inside = contains_xy(inset, G[:, 0], G[:, 1])
            self._grid_key, self._grid_pts = key, G[inside]
        pts = self._grid_pts
        if len(pts) == 0:
            return 1.0
        segs = [(pos[u], pos[v]) for u, v in g.edges if np.linalg.norm(pos[u] - pos[v]) > 1e-6]
        if not segs:
            return 0.0
        A = np.array([a for a, _ in segs])
        B = np.array([b for _, b in segs])
        d = B - A  # [S, 2]
        L2 = (d**2).sum(1) + 1e-12
        # point-to-segment distance for all (pt, seg) pairs
        w = pts[:, None, :] - A[None]  # [P, S, 2]
        t = np.clip((w * d[None]).sum(-1) / L2[None], 0, 1)
        proj = A[None] + t[..., None] * d[None]
        dist = np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(1)
        return float(np.mean(dist <= self.p.shop_depth * 1.1))

    # ---- relaxation ---------------------------------------------------------------------------
    def relax(self, g: nx.Graph, pos: dict[str, np.ndarray], roles: dict[str, str], inset: Polygon, lines: list[LineString], frame: list[float], rng: np.random.RandomState, fixed: set[str] | None = None) -> dict[str, np.ndarray]:
        p = self.p
        fixed = fixed or set()
        nodes = list(g.nodes)
        pos = {k: v.copy() for k, v in pos.items()}
        checker = _NodeCrossChecker(g)
        for it in range(p.iters):
            T = 1.0 - it / p.iters  # annealed step
            disp = {v: np.zeros(2) for v in nodes}
            # role-based attraction
            for v in nodes:
                if roles[v] == "outer":
                    tgt = _boundary_point(inset, pos[v])
                    disp[v] += p.w_boundary * (tgt - pos[v]) * 0.5
                elif roles[v] == "leaf" and g.degree(v) == 1:
                    # a dead-end key point is an entrance (ground floor) – it belongs at the façade side of the inset,
                    # not deep inside; the medial axis only keeps it from drifting sideways
                    tgt = _boundary_point(inset, pos[v])
                    disp[v] += p.w_leaf_facade * (tgt - pos[v]) * 0.5
                    if lines:
                        ta = np.array(nearest_on_lines(lines, tuple(pos[v])))
                        disp[v] += 0.3 * p.w_axis * (ta - pos[v]) * 0.5
                elif roles[v] in ("branch", "leaf") and lines:
                    tgt = np.array(nearest_on_lines(lines, tuple(pos[v])))
                    disp[v] += p.w_axis * (tgt - pos[v]) * 0.5
            # orthogonal + straight edges
            for u, v in g.edges:
                d = pos[v] - pos[u]
                snapped = _snap_angle(d, frame)
                corr = (snapped - d) * 0.5 * p.w_ortho
                disp[v] += corr
                disp[u] -= corr
            for v in nodes:
                if g.degree(v) == 2 and roles[v] in ("branch", "outer", "core"):
                    a, b = list(g.neighbors(v))
                    mid = (pos[a] + pos[b]) / 2
                    # pull towards the segment a–b only if that keeps the corridor straight (not for corner nodes)
                    da, db = pos[a] - pos[v], pos[b] - pos[v]
                    cosang = np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)
                    if cosang < -0.5:  # already roughly straight -> straighten fully
                        disp[v] += p.w_straight * (mid - pos[v]) * 0.3
            # angle opening: two corridors leaving a node at < min_angle form a sharp wedge that no real mall has
            # (a shop cannot fit in it); rotate both far ends apart around the node until the angle opens
            cos_min = np.cos(np.radians(p.min_angle_deg))
            for v in nodes:
                nb = list(g.neighbors(v))
                if len(nb) < 2:
                    continue
                for i in range(len(nb)):
                    for j in range(i + 1, len(nb)):
                        a, b = nb[i], nb[j]
                        da, db = pos[a] - pos[v], pos[b] - pos[v]
                        La, Lb = np.linalg.norm(da), np.linalg.norm(db)
                        if La < 1e-9 or Lb < 1e-9:
                            continue
                        c = np.dot(da, db) / (La * Lb)
                        if c > cos_min:  # angle too small
                            # perpendicular directions pushing a and b apart (rotation about v)
                            na = np.array([-da[1], da[0]]) / La
                            nb_ = np.array([-db[1], db[0]]) / Lb
                            sgn = 1.0 if np.cross(da, db) < 0 else -1.0  # open away from each other
                            k = p.w_angle * (c - cos_min) * min(La, Lb) * 0.5
                            disp[a] += sgn * na * k
                            disp[b] -= sgn * nb_ * k
            # repulsion (vectorised; coincident nodes get a deterministic tiny offset)
            P = np.array([pos[v] for v in nodes])
            D = P[:, None] - P[None]
            dist = np.linalg.norm(D, axis=-1)
            zero = (dist < 1e-6) & ~np.eye(len(P), dtype=bool)
            if zero.any():
                jit = rng.normal(0, 0.1, size=D.shape)
                D = np.where(zero[..., None], jit, D)
                dist = np.linalg.norm(D, axis=-1)
            dist = dist + np.eye(len(P)) * 1e9
            close = dist < p.min_spacing
            unit = D / dist[..., None]
            push = (unit * (p.min_spacing - dist)[..., None] * close[..., None]).sum(1)
            for i, v in enumerate(nodes):
                disp[v] += p.w_repel * push[i] * 0.5
            # edge–node clearance (node too close to a non-incident corridor), vectorised point–segment distances
            if it % 3 == 0 and g.number_of_edges() > 0:
                E_ = list(g.edges)
                A = np.array([pos[a] for a, _ in E_])
                B = np.array([pos[b] for _, b in E_])
                dseg = B - A
                L2 = (dseg**2).sum(1) + 1e-12
                w = P[:, None, :] - A[None]  # [N, S, 2]
                t = np.clip((w * dseg[None]).sum(-1) / L2[None], 0, 1)
                proj = A[None] + t[..., None] * dseg[None]
                away = P[:, None, :] - proj  # [N, S, 2]
                dd = np.linalg.norm(away, axis=-1)
                nid = {v: i for i, v in enumerate(nodes)}
                inc = np.zeros((len(nodes), len(E_)), bool)
                for j, (a, b) in enumerate(E_):
                    inc[nid[a], j] = inc[nid[b], j] = True
                lim = p.min_spacing * 0.6
                hit = (dd < lim) & ~inc & (dd > 1e-9)
                if hit.any():
                    push_e = (away / (dd[..., None] + 1e-12) * ((lim - dd) * hit)[..., None] * 2.5).sum(1)
                    for i, v in enumerate(nodes):
                        disp[v] += push_e[i]
            # apply with planarity check; a move that would create a crossing is retried with a halved step
            # (line search) so crowded Tutte interiors can still expand instead of getting stuck
            Pcur = np.array([pos[v] for v in nodes])
            for i, v in enumerate(nodes):
                if v in fixed:
                    continue
                old = pos[v]
                full = disp[v] * p.step * (0.4 + 0.6 * T)
                if np.linalg.norm(full) < 1e-9:
                    continue
                for frac in (1.0, 0.5, 0.25, 0.125):
                    pos[v] = _project_inside(old + full * frac, inset)
                    Pcur[i] = pos[v]
                    if not checker.crosses(pos, v, Pcur):
                        break
                else:
                    pos[v] = old
                    Pcur[i] = old
        return pos

    def repair_crossings(self, g: nx.Graph, pos: dict[str, np.ndarray], roles: dict[str, str], inset: Polygon, rng: np.random.RandomState, passes: int = 3, n_random: int = 12, fixed: set[str] | None = None) -> dict[str, np.ndarray]:
        """Remove residual crossings left by the initialisation (Tutte on a non-convex inset can cross). For each
        endpoint of a crossing pair (interior nodes first) try: neighbour barycentre, points along the edges to its
        neighbours, random points in the inset; keep the first move that lowers the crossing count."""
        pos = {k: v.copy() for k, v in pos.items()}
        E = list(g.edges)
        for _ in range(passes):
            total = count_crossings(g, pos)
            if total == 0:
                break
            bad: list[str] = []
            cm = np.triu(crossing_matrix(E, pos), 1)
            for i, j in zip(*np.nonzero(cm)):
                bad.extend([*E[i], *E[j]])
            order = sorted((v for v in set(bad) if not fixed or v not in fixed), key=lambda v: (roles[v] == "outer", -bad.count(v)))
            minx, miny, maxx, maxy = inset.bounds
            for v in order:
                cur = count_crossings(g, pos)
                if cur == 0:
                    break
                nb = list(g.neighbors(v))
                cands = []
                if nb:
                    bary = np.mean([pos[u] for u in nb], axis=0)
                    cands.append(bary)
                    for u in nb:
                        for t in (0.3, 0.6):
                            cands.append(pos[v] + (pos[u] - pos[v]) * t)
                for _k in range(n_random):
                    cands.append(np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)]))
                if nb:  # local ring around the neighbour barycentre (small moves first)
                    bary = np.mean([pos[u] for u in nb], axis=0)
                    r0 = 0.5 * float(np.mean([np.linalg.norm(pos[u] - pos[v]) for u in nb]) + 1e-9)
                    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        cands.append(bary + r0 * np.array([np.cos(ang), np.sin(ang)]))
                old = pos[v]
                best = (cur, old)
                for c in cands:
                    c = _project_inside(c, inset)
                    pos[v] = c
                    k = count_crossings(g, pos)
                    if k < best[0]:
                        best = (k, c)
                        if k == 0:
                            break
                pos[v] = best[1]
        return pos

    def open_angles(self, g: nx.Graph, pos: dict[str, np.ndarray], roles: dict[str, str], inset: Polygon, passes: int = 3, fixed: set[str] | None = None) -> dict[str, np.ndarray]:
        """Post-pass: for every sharp wedge (angle < min_angle at node v between neighbours a, b) try moving the
        endpoint that is not on the outer loop (or the shorter arm) sideways so the angle opens to ~min_angle;
        the move is kept only if it stays inside the inset, creates no crossing and does not create a new sharp wedge
        elsewhere. Purely local; the graph is unchanged."""
        p = self.p
        pos = {k: v.copy() for k, v in pos.items()}
        checker = _NodeCrossChecker(g)
        nodes = list(g.nodes)
        nid = {v: i for i, v in enumerate(nodes)}
        target = np.radians(p.min_angle_deg)
        for _ in range(passes):
            moved = 0
            for v in nodes:
                nb = list(g.neighbors(v))
                for i in range(len(nb)):
                    for j in range(i + 1, len(nb)):
                        a, b = nb[i], nb[j]
                        da, db = pos[a] - pos[v], pos[b] - pos[v]
                        La, Lb = np.linalg.norm(da), np.linalg.norm(db)
                        if La < 1e-9 or Lb < 1e-9:
                            continue
                        ang = np.arccos(np.clip(np.dot(da, db) / (La * Lb), -1, 1))
                        if ang >= target:
                            continue
                        # candidate mover: prefer non-outer, then the shorter arm
                        order = sorted([n for n in (a, b) if not fixed or n not in fixed], key=lambda n: (roles.get(n) == "outer", np.linalg.norm(pos[n] - pos[v])))
                        before = sharp_angle_rate(g, pos, p.min_angle_deg)
                        for m in order:
                            other = b if m == a else a
                            dm, do = pos[m] - pos[v], pos[other] - pos[v]
                            Lm = np.linalg.norm(dm)
                            # rotate the arm v->m away from v->other to reach the target angle (try both senses)
                            base = np.arctan2(do[1], do[0])
                            cur = np.arctan2(dm[1], dm[0])
                            sgn = 1.0 if ((cur - base + np.pi) % (2 * np.pi) - np.pi) >= 0 else -1.0
                            for frac in (1.0, 0.75, 0.5):
                                new_ang = base + sgn * (target * frac + (1 - frac) * ang)
                                cand = pos[v] + np.array([np.cos(new_ang), np.sin(new_ang)]) * Lm
                                cand = _project_inside(cand, inset)
                                old = pos[m]
                                pos[m] = cand
                                Pcur = np.array([pos[n] for n in nodes])
                                ok = not checker.crosses(pos, m, Pcur) and sharp_angle_rate(g, pos, p.min_angle_deg) < before
                                if ok:
                                    moved += 1
                                    break
                                pos[m] = old
                            else:
                                continue
                            break
            if moved == 0:
                break
        return pos

    def snap_corners(self, g: nx.Graph, pos: dict[str, np.ndarray], roles: dict[str, str], inset: Polygon, fixed: set[str] | None = None) -> dict[str, np.ndarray]:
        corners = np.array(inset.exterior.coords[:-1])
        pos = {k: v.copy() for k, v in pos.items()}
        used = set()
        for v in g.nodes:
            if roles[v] != "outer" or (fixed and v in fixed):
                continue
            d = np.linalg.norm(corners - pos[v], axis=1)
            i = int(d.argmin())
            if d[i] < self.p.snap_dist and i not in used:
                cand = corners[i] + (inset.centroid.coords[0] - corners[i]) * 0.02
                old = pos[v]
                pos[v] = cand
                if any(_seg_cross(pos[v], pos[u], pos[a], pos[b]) for u in g.neighbors(v) for a, b in g.edges if v not in (a, b) and u not in (a, b)):
                    pos[v] = old
                else:
                    used.add(i)
        return pos

    # ---- main -----------------------------------------------------------------------------------
    def fit(self, topology: TopologyGraph, outline: Outline, seed: int = 0, skeleton_nodes: set[str] | None = None, anchors: dict[str, tuple[float, float]] | None = None) -> FitResult:
        """``skeleton_nodes`` (Stage-1 prototype): its cycle becomes the outer loop pinned to the inset boundary.

        ``anchors`` (renovation mode): real positions (metres, outline frame) of nodes that must **not move** – typically
        the existing mall's skeleton key points. Only the new nodes are placed (Tutte between their anchored neighbours,
        then relaxed), so the core corridors keep their built shape and the new corridors are grafted onto them."""
        p = self.p
        g = to_networkx(topology)
        rng = np.random.RandomState(seed)
        depth = effective_depth(outline, p.shop_depth, p.depth_frac, p.depth_area_coef)
        inset = inset_region(outline, depth)
        fixed_pos = {v: np.asarray(xy, float) for v, xy in sorted((anchors or {}).items()) if v in g.nodes}
        fixed = set(fixed_pos)
        g = nx.Graph(); g.add_nodes_from(sorted(to_networkx(topology).nodes)); g.add_edges_from(sorted(tuple(sorted(e)) for e in to_networkx(topology).edges))  # deterministic order (str hashing is randomised per process)
        # spacing adapts to how many key points must share the inset (never above the configured value)
        self._spacing_backup = p.min_spacing
        p.min_spacing = float(min(p.min_spacing, 0.85 * np.sqrt(inset.area / max(g.number_of_nodes(), 1))))
        _, lines = medial_axis_graph(outline, depth, px=p.raster_px)
        frame = dominant_directions(outline.polygon)
        _, info = planar_corridor_embedding(topology, PlanarEmbedParams(ortho_weight=0.0, relax_iters=0), skeleton_nodes=skeleton_nodes)
        roles = _roles(g, info)
        if fixed:
            cands = _init_anchored(g, fixed_pos, inset, lines, depth, rng, n=max(p.n_restarts, 1), jitter=p.anchor_jitter * p.min_spacing)
        else:
            cands = _init_on_inset(g, roles, list(info.get("outer_cycle", [])), inset, lines, depth, p.snap_dist, p.n_offsets)
        scored = []
        for pos0 in cands:
            s, _ = self.score(g, pos0, roles, inset, lines, frame, fixed)
            scored.append((s, pos0))
        scored.sort(key=lambda t: t[0])
        # seed 0 = deterministic best-first; other seeds sample ``n_restarts`` poses from the top-2·n pool so a
        # designer's "try again" really explores a different loop offset / direction (label-free, all plausible)
        pool = scored[: p.n_restarts] if not seed else scored[: max(p.n_restarts, 2 * p.n_restarts)]
        if seed and len(pool) > p.n_restarts:
            idx = rng.choice(len(pool), size=p.n_restarts, replace=False)
            pool = [pool[i] for i in sorted(idx)]
        best = None
        for s0, pos0 in pool:
            pos0 = self.repair_crossings(g, pos0, roles, inset, rng, fixed=fixed)  # planar start (relax never introduces crossings)
            pos1 = self.relax(g, pos0, roles, inset, lines, frame, rng, fixed=fixed)
            pos1 = self.snap_corners(g, pos1, roles, inset, fixed=fixed)
            pos1 = self.open_angles(g, pos1, roles, inset, fixed=fixed)
            if count_crossings(g, pos1):
                pos1 = self.repair_crossings(g, pos1, roles, inset, rng, passes=6 if fixed else 3, n_random=48 if fixed else 12, fixed=fixed)
                if fixed and count_crossings(g, pos1):  # anchored edges cannot move: re-relax the new nodes from the repaired start
                    pos1 = self.relax(g, pos1, roles, inset, lines, frame, rng, fixed=fixed)
                    pos1 = self.repair_crossings(g, pos1, roles, inset, rng, passes=6, n_random=48, fixed=fixed)
            s, diag = self.score(g, pos1, roles, inset, lines, frame, fixed)
            if best is None or s < best[0]:
                best = (s, pos1, diag)
        s, pos, diag = best  # type: ignore[misc]
        diag["min_spacing_m"] = p.min_spacing
        p.min_spacing = self._spacing_backup
        diag.update({"n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(), "n_anchored": len(fixed), "roles": {r: sum(1 for v in roles.values() if v == r) for r in ("outer", "core", "branch", "leaf")}, "inset_area_m2": float(inset.area), "outline_area_m2": outline.area, "effective_depth_m": depth, "n_candidates": len(cands)})
        return FitResult(positions={k: (float(v[0]), float(v[1])) for k, v in pos.items()}, roles=roles, inset=inset, axis_lines=lines, frame_angles=frame, score=float(s), diagnostics=diag)


class _NodeCrossChecker:
    """Incremental planarity test: does moving ``v`` make any incident edge cross a non-adjacent edge?
    Per-node index structures are built once; each query is one vectorised orientation test over the
    edges not touching ``v`` or the incident edge's other endpoint."""

    def __init__(self, g: nx.Graph) -> None:
        self.E = list(g.edges)
        self.nodes = list(g.nodes)
        self.nid = {v: i for i, v in enumerate(self.nodes)}
        ea = np.array([self.nid[a] for a, _ in self.E])
        eb = np.array([self.nid[b] for _, b in self.E])
        self.ea, self.eb = ea, eb
        self.nbrs = {v: list(g.neighbors(v)) for v in self.nodes}
        # for each node: mask of edges not incident to it
        self.not_inc = {v: (ea != self.nid[v]) & (eb != self.nid[v]) for v in self.nodes}

    def crosses(self, pos: dict[str, np.ndarray], v: str, P: np.ndarray | None = None) -> bool:
        """``P`` (optional) = current positions as an [N, 2] array in ``self.nodes`` order with ``pos[v]`` already
        written into row ``nid[v]``; avoids rebuilding the array from the dict on every query."""
        nb = self.nbrs[v]
        if not nb:
            return False
        if P is None:
            P = np.array([pos[n] for n in self.nodes])
        sel = self.not_inc[v]
        A, B = P[self.ea[sel]], P[self.eb[sel]]  # other edges [S, 2]
        if len(A) == 0:
            return False
        ea_s, eb_s = self.ea[sel], self.eb[sel]
        p1 = P[self.nid[v]]
        U = np.array([self.nid[u] for u in nb])
        p2 = P[U]  # [K, 2] incident-edge far ends
        # orientation tests broadcast over incident edges (K) x other edges (S)
        d = p2 - p1  # [K, 2]
        o1 = d[:, None, 0] * (A[None, :, 1] - p1[1]) - d[:, None, 1] * (A[None, :, 0] - p1[0])
        o2 = d[:, None, 0] * (B[None, :, 1] - p1[1]) - d[:, None, 1] * (B[None, :, 0] - p1[0])
        e = B - A  # [S, 2]
        o3 = e[None, :, 0] * (p1[1] - A[None, :, 1]) - e[None, :, 1] * (p1[0] - A[None, :, 0])
        o4 = e[None, :, 0] * (p2[:, None, 1] - A[None, :, 1]) - e[None, :, 1] * (p2[:, None, 0] - A[None, :, 0])
        cross = (o1 * o2 < 0) & (o3 * o4 < 0)
        # an "other" edge that touches the far end u is adjacent to (v,u): never a proper crossing
        adj = (ea_s[None, :] == U[:, None]) | (eb_s[None, :] == U[:, None])
        return bool(np.any(cross & ~adj))


def _node_crosses(g: nx.Graph, pos: dict[str, np.ndarray], v: str, E: list[tuple[str, str]]) -> bool:  # kept for API compatibility
    return _NodeCrossChecker(g).crosses(pos, v)


def _seg_cross(p1, p2, p3, p4) -> bool:  # noqa: ANN001
    def orient(a, b, c):  # noqa: ANN001, ANN202
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


__all__ = ["CorridorFitter", "FitParams", "FitResult"]
