"""Matplotlib rendering of layouts and topologies (PNG)."""

from __future__ import annotations

from pathlib import Path

from mall_space_planner.schemas import GeneratedLayout, TopologyGraph

_COLORS = {"shop": "#cfe8ff", "anchor": "#dcd6f7", "corridor": "#f0c987", "atrium": "#b5e7a0", "junction": "#222222", "entrance": "#d9480f"}


def render_layout_png(layout: GeneratedLayout, path: str | Path, dpi: int = 150, title: str | None = None, show_labels: bool = True) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(MplPolygon(layout.boundary.exterior, closed=True, fill=True, facecolor="#fafafa", edgecolor="#333", linewidth=2))
    for h in layout.boundary.holes:
        ax.add_patch(MplPolygon(h, closed=True, fill=True, facecolor="white", edgecolor="#333"))
    for u in layout.units:
        if u.polygon:
            ax.add_patch(MplPolygon(u.polygon, closed=True, facecolor=_COLORS.get(u.kind, "#ddd"), edgecolor="#3a7bd5" if u.kind == "shop" else "#666", linewidth=0.6, alpha=0.9))
            if show_labels and u.kind == "shop" and u.area and u.area > 80:
                ax.text(u.centroid[0], u.centroid[1], f"{u.area:.0f}", fontsize=5, ha="center", va="center", color="#234")
        elif u.kind == "entrance" and u.centroid:
            ax.plot(*u.centroid, marker="^", color=_COLORS["entrance"], markersize=12)
    pos = layout.skeleton_positions
    for a, b in layout.topology.edges():
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#d9480f", linewidth=2.2)
    for n, p in pos.items():
        ax.plot(p[0], p[1], "o", color=_COLORS["junction"], markersize=4)
        if show_labels:
            ax.text(p[0], p[1], n, fontsize=6, color="#111")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(title or f"{layout.layout_id} · prototype {layout.prototype_id} · {sum(1 for u in layout.units if u.kind == 'shop')} shops")
    ax.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def render_topology_png(graph: TopologyGraph, path: str | Path, highlight: set[str] | None = None, title: str | None = None, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    from mall_space_planner.topology.convert import to_networkx

    g = to_networkx(graph)
    pos = {n: graph.positions[n] for n in graph.nodes if n in graph.positions}
    if len(pos) != len(graph.nodes):
        pos = nx.spring_layout(g, seed=seed, pos=pos or None, fixed=list(pos) or None)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#d9480f" if highlight and n in highlight else "#3a7bd5" for n in g.nodes]
    nx.draw_networkx(g, pos=pos, ax=ax, node_color=colors, node_size=120, font_size=6, edge_color="#777")
    ax.set_title(title or f"{graph.num_nodes} nodes / {graph.num_edges} edges")
    ax.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
