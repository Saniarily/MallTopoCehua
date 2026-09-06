"""Pretty topology drawing for thesis figures.

Layout strategy (avoids the "skeleton clump" problem):
1. Skeleton positions: Kamada–Kawai on the *skeleton alone* (spreads it like a plan), then a
   min-distance repulsion pass so no two skeleton nodes touch.
2. New nodes: initialised next to their attached skeleton/new neighbour, relaxed by a spring layout
   with skeleton fixed, then a repulsion pass over *all* nodes (skeleton still fixed) so new nodes
   never overlap each other or the skeleton.
3. Everything is normalised to a unit square; node sizes are computed in *data units* via a scaled
   marker so overlap is impossible by construction (node radius < half the min distance).

Skeleton positions are computed once per example and shared across panels (skeleton / generated /
ground truth) so the reader compares like with like.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from mall_space_planner.reporting.style import load_style
from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.topology.convert import to_networkx


# --------------------------------------------------------------------------- layout
def _normalise(pos: dict) -> dict:
    P = np.array(list(pos.values()), float)
    lo, hi = P.min(0), P.max(0)
    span = max((hi - lo).max(), 1e-9)
    return {k: ((v[0] - lo[0]) / span, (v[1] - lo[1]) / span) for k, v in pos.items()}


def _repel(pos: dict, min_dist: float, fixed: set, iters: int = 200) -> dict:
    keys = list(pos)
    P = np.array([pos[k] for k in keys], float)
    fx = np.array([k in fixed for k in keys])
    for _ in range(iters):
        d = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        over = np.clip(min_dist - dist, 0, None)
        if over.max() <= 1e-6:
            break
        unit = d / np.where(dist[:, :, None] == np.inf, 1, dist[:, :, None] + 1e-9)
        step = (unit * over[:, :, None]).sum(1) * 0.5
        # if both are fixed nothing moves; if one is fixed the other takes the whole displacement
        both_free = (~fx)[:, None] & (~fx)[None, :]
        one_fixed = (~fx)[:, None] & fx[None, :]
        step = (unit * over[:, :, None] * (0.5 * both_free + 1.0 * one_fixed)[:, :, None]).sum(1)
        step[fx] = 0
        P += step
    return {k: tuple(P[i]) for i, k in enumerate(keys)}


def skeleton_layout(skeleton: nx.Graph, seed: int = 0) -> dict:
    """Spread-out positions for a skeleton (unit square, min spacing enforced)."""
    n = skeleton.number_of_nodes()
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(skeleton.nodes)): (0.5, 0.5)}
    if n == 2:
        a, b = list(skeleton.nodes)
        return {a: (0.2, 0.5), b: (0.8, 0.5)}
    try:
        pos = nx.kamada_kawai_layout(skeleton) if nx.is_connected(skeleton) else nx.spring_layout(skeleton, seed=seed, iterations=300, k=1.6 / np.sqrt(n))
    except Exception:  # noqa: BLE001
        pos = nx.spring_layout(skeleton, seed=seed, iterations=300)
    pos = _normalise(pos)
    md = 0.75 / np.sqrt(n)
    return _normalise(_repel(pos, min_dist=md, fixed=set()))


def full_layout(full: nx.Graph, skeleton_nodes: set, sk_pos: dict, seed: int = 0) -> dict:
    """Positions for the full graph keeping ``sk_pos`` for skeleton nodes."""
    n = full.number_of_nodes()
    if n == 0:
        return {}
    rng = np.random.RandomState(seed)
    init = {k: sk_pos[k] for k in full.nodes if k in sk_pos}
    # BFS from skeleton to place new nodes progressively next to an already placed neighbour
    order = [v for v in full.nodes if v not in init]
    placed = set(init)
    pending = list(order)
    guard = 0
    while pending and guard < 10 * len(order) + 10:
        guard += 1
        v = pending.pop(0)
        nb = [u for u in full.neighbors(v) if u in placed]
        if not nb and pending:
            pending.append(v)
            continue
        if nb:
            c = np.mean([init[u] for u in nb], 0)
            # push outward from centre of the skeleton so branches grow away from it
            centre = np.mean(list(sk_pos.values()), 0) if sk_pos else np.array([0.5, 0.5])
            out = c - centre
            out = out / (np.linalg.norm(out) + 1e-9)
            init[v] = tuple(c + out * 0.10 + rng.normal(0, 0.03, 2))
        else:
            init[v] = tuple(rng.uniform(0, 1, 2))
        placed.add(v)
    fixed = [k for k in full.nodes if k in sk_pos]
    k = 0.55 / np.sqrt(n)
    try:
        pos = nx.spring_layout(full, pos=init, fixed=fixed or None, k=k, iterations=60, seed=seed)
    except Exception:  # noqa: BLE001
        pos = init
    pos = {a: tuple(map(float, b)) for a, b in pos.items()}
    # clamp new nodes into a ring around the skeleton so the skeleton stays legible (never shrinks to a dot)
    if fixed:
        S = np.array([sk_pos[v] for v in fixed])
        c = S.mean(0)
        R = max(np.linalg.norm(S - c, axis=1).max(), 0.25)
        lim = 1.35 * R
        for v in full.nodes:
            if v in sk_pos:
                continue
            d = np.array(pos[v]) - c
            r = np.linalg.norm(d)
            if r > lim:
                pos[v] = tuple(c + d / r * lim)
    md = 0.62 / np.sqrt(n)
    pos = _repel(pos, min_dist=md, fixed=set(fixed), iters=300)
    return pos


# --------------------------------------------------------------------------- drawing
def draw_topology(ax, graph: TopologyGraph | nx.Graph, skeleton_nodes: set | None = None, *, sk_pos: dict | None = None, pos: dict | None = None, title: str = "", highlight_edges=None, labels: bool = False, seed: int = 0, node_scale: float = 1.0, size_ref_n: int | None = None, frame: tuple | None = None, title_loc: str = "center", title_pad: float = 4) -> dict:
    """Draw one topology. Provide ``sk_pos`` (shared skeleton positions) or full ``pos``.

    Returns the positions used so callers can reuse them for the next panel.
    """
    s = load_style()
    g = graph if isinstance(graph, nx.Graph) else to_networkx(graph)
    sk = set(skeleton_nodes or [])
    if pos is None:
        if sk_pos is None:
            sk_pos = skeleton_layout(g.subgraph([v for v in g.nodes if v in sk]) if sk else g, seed) if sk else None
        if sk:
            pos = full_layout(g, sk, sk_pos, seed)
        else:
            pos = skeleton_layout(g, seed)
    # node radius in data units: a bit below half the min distance used by repulsion.
    # ``size_ref_n`` lets a row of panels (skeleton / generated / truth) share one node size.
    n = max(g.number_of_nodes(), 1)
    n_ref = max(size_ref_n or n, 1)
    P = np.array(list(pos.values()), float)
    r_data = 0.62 / np.sqrt(n_ref) * 0.42 * node_scale
    pad = r_data * 2.2
    if frame is not None:
        (x0, x1), (y0, y1) = frame
    else:
        x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    # Equal aspect *without* letting matplotlib shrink the axes box (which mis-aligns titles) and
    # without ``adjustable="datalim"`` (which is ignored under bbox_inches="tight" and clips the
    # drawing): widen the data limits ourselves to the axes-box aspect ratio.
    ax.axis("off")
    ax.figure.canvas.draw()  # to get axis size in pixels
    bbox = ax.get_window_extent()
    box_aspect = bbox.height / max(bbox.width, 1e-9)
    w, h = x1 - x0, y1 - y0
    if h / max(w, 1e-9) < box_aspect:  # box is taller than the data -> grow y
        extra = w * box_aspect - h
        y0, y1 = y0 - extra / 2, y1 + extra / 2
    else:  # box is wider -> grow x
        extra = h / box_aspect - w
        x0, x1 = x0 - extra / 2, x1 + extra / 2
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    px_per_data = bbox.width / (x1 - x0)
    pt_per_px = 72.0 / ax.figure.dpi
    r_pt = r_data * px_per_data * pt_per_px
    size_sk = (2 * r_pt) ** 2
    size_new = (2 * r_pt * 0.85) ** 2
    ours, dark, acc = s["palette"]["ours"], "#2B2B2B", s["palette"]["highlight"]
    sk_e = [e for e in g.edges if e[0] in sk and e[1] in sk]
    new_e = [e for e in g.edges if not (e[0] in sk and e[1] in sk)]
    lw_sk, lw_new = (2.0, 1.1) if n <= 40 else (1.5, 0.9)
    nx.draw_networkx_edges(g, pos, edgelist=new_e, ax=ax, width=lw_new, edge_color=ours, alpha=0.8)
    nx.draw_networkx_edges(g, pos, edgelist=sk_e, ax=ax, width=lw_sk, edge_color=dark, alpha=0.95)
    if highlight_edges:
        nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, ax=ax, width=lw_sk + 1.0, edge_color=acc)
    new_n = [v for v in g.nodes if v not in sk]
    sk_n = [v for v in g.nodes if v in sk]
    if new_n:
        ax.scatter([pos[v][0] for v in new_n], [pos[v][1] for v in new_n], s=size_new, facecolors="white", edgecolors=ours, linewidths=1.2, zorder=3)
    if sk_n:
        ax.scatter([pos[v][0] for v in sk_n], [pos[v][1] for v in sk_n], s=size_sk, facecolors=dark, edgecolors="white", linewidths=0.9, zorder=4)
    if labels:
        for v in g.nodes:
            ax.text(pos[v][0], pos[v][1], v, ha="center", va="center", fontsize=max(4, min(7, r_pt * 0.9)), color="white" if v in sk else dark, zorder=5)
    if title:
        ax.set_title(title, fontsize=s["fonts"]["size_annot"] + 0.5, loc=title_loc, pad=title_pad)
    return pos


def frame_of(*pos_dicts: dict) -> tuple:
    """Union bounding box of several position dicts -> ``((x0, x1), (y0, y1))`` for ``draw_topology(frame=...)``."""
    P = np.array([xy for d in pos_dicts for xy in d.values()], float)
    return (P[:, 0].min(), P[:, 0].max()), (P[:, 1].min(), P[:, 1].max())


def legend_handles():
    from matplotlib.lines import Line2D

    s = load_style()
    return [
        Line2D([], [], color="#2B2B2B", marker="o", lw=2.0, markersize=7, markeredgecolor="white", label="骨架节点 / 连接（阶段一原型，生成中保持不变）"),
        Line2D([], [], color=s["palette"]["ours"], marker="o", markerfacecolor="white", lw=1.1, markersize=6, label="新增空间单元 / 连接"),
    ]
