"""Outline geometry used by the fitter: inset region and medial-axis (centre-line) graph.

* ``inset_region(outline, depth)`` – the outline shrunk by one shop depth; corridors live inside it and
  its boundary is where the *outer corridor loop* wants to be (R1 in ``docs/methodology.md`` §Stage 3).
* ``medial_axis_graph(outline, depth, px)`` – skeletonise the inset region on a raster (skimage), turn the
  skeleton pixels into a graph, prune short spurs, and return (networkx graph with ``pos`` in metres,
  list of polylines). Wings of the outline become single centre-lines; the main body becomes a loop or
  a spine. Branch nodes of this graph are natural candidates for corridor junctions (M nodes).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from mall_space_planner.stage3.outline import Outline


def _largest(geom) -> Polygon:  # noqa: ANN001
    if isinstance(geom, Polygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return max(polys, key=lambda g: g.area) if polys else Polygon()


def inset_region(outline: Outline, depth: float) -> Polygon:
    """Outline shrunk by ``depth`` (m). Falls back to a smaller inset if the outline is too thin."""
    poly = outline.polygon
    for d in (depth, depth * 0.75, depth * 0.5, depth * 0.25, 0.0):
        g = poly.buffer(-d, join_style="mitre").buffer(0)
        g = _largest(g)
        if not g.is_empty and g.area > 0.15 * poly.area:
            return g
    return poly


def _rasterise(poly: Polygon, px: float) -> tuple[np.ndarray, tuple[float, float]]:
    from skimage.draw import polygon as sk_polygon

    minx, miny, maxx, maxy = poly.bounds
    W, H = int(np.ceil((maxx - minx) / px)) + 3, int(np.ceil((maxy - miny) / px)) + 3
    img = np.zeros((H, W), bool)
    xs, ys = np.array(poly.exterior.coords).T
    rr, cc = sk_polygon((ys - miny) / px + 1, (xs - minx) / px + 1, img.shape)
    img[rr, cc] = True
    for ring in poly.interiors:
        xs, ys = np.array(ring.coords).T
        rr, cc = sk_polygon((ys - miny) / px + 1, (xs - minx) / px + 1, img.shape)
        img[rr, cc] = False
    return img, (minx - px, miny - px)


def medial_axis_graph(outline: Outline, depth: float, px: float = 1.0, min_spur: float | None = None) -> tuple[nx.Graph, list[LineString]]:
    """Centre-line graph of the inset region. Node attribute ``pos`` = (x, y) m; edge attribute ``length``."""
    from skimage.morphology import skeletonize

    region = inset_region(outline, depth)
    img, (ox, oy) = _rasterise(region, px)
    sk = skeletonize(img)
    ys, xs = np.nonzero(sk)
    if len(xs) == 0:
        g = nx.Graph()
        c = region.centroid
        g.add_node(0, pos=(c.x, c.y))
        return g, []
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs, strict=True))}
    g = nx.Graph()
    for (y, x), i in idx.items():
        g.add_node(i, pos=(ox + x * px, oy + y * px))
    for (y, x), i in idx.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in idx:
                    j = idx[(y + dy, x + dx)]
                    if i < j:
                        g.add_edge(i, j, length=float(np.hypot(dx, dy) * px))
    # collapse pixel chains into polylines between branch / end nodes
    g = _collapse_chains(g)
    min_spur = min_spur if min_spur is not None else max(2.0 * depth, 6.0)
    g = _prune_spurs(g, min_spur)
    lines = [LineString(d["pts"]) for _, _, d in g.edges(data=True) if len(d.get("pts", [])) >= 2]
    return g, lines


def _collapse_chains(g: nx.Graph) -> nx.Graph:
    """Pixel graph -> graph whose nodes are branch/end points and edges carry the polyline ``pts``."""
    key = [n for n in g.nodes if g.degree(n) != 2]
    if not key:  # a pure cycle: pick one node as key
        key = [next(iter(g.nodes))]
    keyset = set(key)
    h = nx.Graph()
    for n in key:
        h.add_node(n, pos=g.nodes[n]["pos"])
    seen = set()
    for n in key:
        for m in g.neighbors(n):
            if (n, m) in seen:
                continue
            pts = [g.nodes[n]["pos"]]
            prev, cur = n, m
            while cur not in keyset:
                pts.append(g.nodes[cur]["pos"])
                nb = [q for q in g.neighbors(cur) if q != prev]
                if not nb:
                    break
                prev, cur = cur, nb[0]
            pts.append(g.nodes[cur]["pos"])
            seen.add((cur, prev))
            if cur in keyset and (n != cur or len(pts) > 2):
                L = float(sum(np.hypot(*(np.subtract(pts[i + 1], pts[i]))) for i in range(len(pts) - 1)))
                if h.has_edge(n, cur):
                    if L < h[n][cur]["length"]:
                        h[n][cur].update(length=L, pts=pts)
                else:
                    h.add_edge(n, cur, length=L, pts=pts)
    return h


def _prune_spurs(g: nx.Graph, min_len: float, rounds: int = 3) -> nx.Graph:
    """Remove short dead-end branches (raster artefacts along the boundary)."""
    g = g.copy()
    for _ in range(rounds):
        leaves = [n for n in g.nodes if g.degree(n) == 1]
        removed = False
        for n in leaves:
            if n not in g:
                continue
            m = next(iter(g.neighbors(n)))
            if g[n][m]["length"] < min_len and g.degree(m) >= 3:
                g.remove_node(n)
                removed = True
        if not removed:
            break
        # merge degree-2 nodes created by pruning
        for n in [n for n in g.nodes if g.degree(n) == 2]:
            a, b = list(g.neighbors(n))
            if a == b or g.has_edge(a, b):
                continue
            pa, pb = g[a][n]["pts"], g[n][b]["pts"]
            if pa[0] != g.nodes[a]["pos"]:
                pa = pa[::-1]
            if pb[0] != g.nodes[n]["pos"]:
                pb = pb[::-1]
            g.add_edge(a, b, length=g[a][n]["length"] + g[n][b]["length"], pts=pa + pb[1:])
            g.remove_node(n)
    return g


def nearest_on_lines(lines: list[LineString], p: tuple[float, float]) -> tuple[float, float]:
    """Closest point on the medial axis to ``p`` (metres)."""
    if not lines:
        return p
    pt = Point(p)
    best = min(lines, key=lambda l: l.distance(pt))
    q = best.interpolate(best.project(pt))
    return (float(q.x), float(q.y))


def boundary_direction(poly: Polygon, p: tuple[float, float]) -> np.ndarray:
    """Unit tangent of the polygon boundary at the point closest to ``p`` (used for orthogonal frames)."""
    ext = poly.exterior
    s = ext.project(Point(p))
    a, b = ext.interpolate(max(s - 0.5, 0)), ext.interpolate(min(s + 0.5, ext.length))
    d = np.array([b.x - a.x, b.y - a.y])
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def dominant_directions(poly: Polygon, k: int = 2) -> list[float]:
    """Principal wall directions (radians in [0, π)) weighted by edge length – the frame corridors snap to."""
    P = np.array(poly.exterior.coords)
    d = np.diff(P, axis=0)
    L = np.linalg.norm(d, axis=1)
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)
    # histogram over 5° bins folded to [0, π/2) (walls are mostly orthogonal)
    fold = np.mod(ang, np.pi / 2)
    bins = np.linspace(0, np.pi / 2, 19)
    h, _ = np.histogram(fold, bins=bins, weights=L)
    order = np.argsort(h)[::-1]
    out = []
    for i in order:
        c = (bins[i] + bins[i + 1]) / 2
        if all(abs(c - o) > np.deg2rad(15) for o in out):
            out.append(float(c))
        if len(out) >= k:
            break
    return out


__all__ = ["boundary_direction", "dominant_directions", "inset_region", "medial_axis_graph", "nearest_on_lines"]
