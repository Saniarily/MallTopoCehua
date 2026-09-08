"""Exporters for workbench candidates (JSON / GeoJSON / SVG / PNG) and a shared matplotlib renderer used by the UI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import mapping

from mall_space_planner.topology.convert import to_networkx

MAIN, SEC, ATRIUM, ENT, CORE = "#F0C987", "#F7DDB0", "#B5E7A0", "#D9480F", "#9e9e9e"


def _polys(geom):  # noqa: ANN001, ANN202
    return [p for p in getattr(geom, "geoms", [geom]) if not p.is_empty]


def candidate_to_json(c) -> dict[str, Any]:  # noqa: ANN001
    return {
        "seed": c.seed,
        "topology": {"nodes": list(c.topology.nodes), "edges": [list(e) for e in c.topology.edges()]},
        "positions": {k: [float(v[0]), float(v[1])] for k, v in c.fit.positions.items()},
        "roles": c.fit.roles,
        "edge_class": {f"{u}--{v}": cls for (u, v), cls in c.plan.edge_class.items()},
        "outline": [list(map(float, p)) for p in np.asarray(c.outline.polygon.exterior.coords)] if c.outline is not None else None,
        "entrances": [{"node": e["node"], "point": [float(e["point"][0]), float(e["point"][1])], "kind": e.get("kind")} for e in c.plan.entrances],
        "vertical_cores": [{"node": v["node"], "point": [float(v["point"][0]), float(v["point"][1])]} for v in c.plan.vertical_cores],
        "atria_area_m2": [float(a.area) for a in c.plan.atria],
        "diagnostics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in c.plan.diagnostics.items()},
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in c.metrics.items() if not isinstance(v, dict)},
        "indicators": c.indicators,
        "pred_score": c.pred_score,
    }


def candidate_to_geojson(c) -> dict[str, Any]:  # noqa: ANN001
    feats = []
    if c.outline is not None:
        feats.append({"type": "Feature", "properties": {"kind": "outline"}, "geometry": mapping(c.outline.polygon)})
    for p in _polys(c.plan.corridors_main):
        feats.append({"type": "Feature", "properties": {"kind": "corridor", "class": "main", "width_m": c.plan.diagnostics.get("main_width_m")}, "geometry": mapping(p)})
    for p in _polys(c.plan.corridors_secondary):
        feats.append({"type": "Feature", "properties": {"kind": "corridor", "class": "secondary", "width_m": c.plan.diagnostics.get("secondary_width_m")}, "geometry": mapping(p)})
    for a in c.plan.atria:
        feats.append({"type": "Feature", "properties": {"kind": "atrium", "area_m2": float(a.area)}, "geometry": mapping(a)})
    for e in c.plan.entrances:
        if not e["stub"].is_empty:
            feats.append({"type": "Feature", "properties": {"kind": "entrance", "node": e["node"], "entrance_kind": e.get("kind")}, "geometry": mapping(e["stub"])})
        feats.append({"type": "Feature", "properties": {"kind": "entrance_point", "node": e["node"]}, "geometry": {"type": "Point", "coordinates": [float(e["point"][0]), float(e["point"][1])]}})
    for v in c.plan.vertical_cores:
        feats.append({"type": "Feature", "properties": {"kind": "vertical_core", "node": v["node"]}, "geometry": mapping(v["polygon"])})
    P = c.fit.positions
    for u, v in c.topology.edges():
        if u in P and v in P:
            feats.append({"type": "Feature", "properties": {"kind": "centerline", "class": c.plan.edge_class.get((u, v), c.plan.edge_class.get((v, u)))}, "geometry": {"type": "LineString", "coordinates": [list(map(float, P[u])), list(map(float, P[v]))]}})
    for n, xy in P.items():
        feats.append({"type": "Feature", "properties": {"kind": "keypoint", "node": n, "role": c.fit.roles.get(n)}, "geometry": {"type": "Point", "coordinates": [float(xy[0]), float(xy[1])]}})
    return {"type": "FeatureCollection", "features": feats, "crs_note": "local metric frame (m)"}


def candidate_to_svg(c, size: int = 900) -> str:  # noqa: ANN001
    poly = c.outline.polygon if c.outline is not None else c.plan.corridor_union
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    s = (size - 20) / max(w, h, 1e-9)

    def pt(x, y):  # noqa: ANN001, ANN202
        return f"{(x - minx) * s + 10:.1f},{(maxy - y) * s + 10:.1f}"

    def path(p, fill, stroke, sw=0.8):  # noqa: ANN001, ANN202
        d = "M " + " L ".join(pt(x, y) for x, y in np.asarray(p.exterior.coords)) + " Z"
        for r in p.interiors:
            d += " M " + " L ".join(pt(x, y) for x, y in np.asarray(r.coords)) + " Z"
        return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" fill-rule="evenodd"/>'

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(w * s) + 20}" height="{int(h * s) + 20}" viewBox="0 0 {int(w * s) + 20} {int(h * s) + 20}">']
    if c.outline is not None:
        out.append(path(c.outline.polygon, "#f4f4f4", "#333", 1.2))
    out += [path(p, SEC, "#b07a2a") for p in _polys(c.plan.corridors_secondary)]
    out += [path(p, MAIN, "#b07a2a") for p in _polys(c.plan.corridors_main)]
    out += [path(e["stub"], MAIN, "#b07a2a") for e in c.plan.entrances if not e["stub"].is_empty]
    out += [path(a, ATRIUM, "#5a9a4a") for a in c.plan.atria]
    out += [path(v["polygon"], CORE, "#555") for v in c.plan.vertical_cores]
    for e in c.plan.entrances:
        x, y = e["point"]
        out.append(f'<circle cx="{pt(x, y).split(",")[0]}" cy="{pt(x, y).split(",")[1]}" r="5" fill="{ENT}"/>')
    out.append("</svg>")
    return "\n".join(out)


def draw_candidate(ax, c, show_network: bool = True, show_plan: bool = True, title: str | None = None) -> None:  # noqa: ANN001
    """Matplotlib rendering shared by PNG export and the Streamlit UI."""
    from matplotlib.patches import Polygon as MplPolygon

    def poly(geom, **kw):  # noqa: ANN001, ANN202
        for p in _polys(geom):
            ax.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, **kw))
            for r in p.interiors:
                ax.add_patch(MplPolygon(np.array(r.coords), closed=True, fc="white", ec=kw.get("ec", "none"), lw=kw.get("lw", 0.5)))

    if c.outline is not None:
        poly(c.outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
    if show_plan:
        poly(c.plan.corridors_secondary, fc=SEC, ec="#b07a2a", lw=0.4)
        poly(c.plan.corridors_main, fc=MAIN, ec="#b07a2a", lw=0.5)
        for a in c.plan.atria:
            poly(a, fc=ATRIUM, ec="#5a9a4a", lw=0.5)
        for e in c.plan.entrances:
            poly(e["stub"], fc=MAIN, ec="#b07a2a", lw=0.4)
            ax.scatter([e["point"][0]], [e["point"][1]], marker="v", s=60, c=ENT, zorder=6)
        for v in c.plan.vertical_cores:
            poly(v["polygon"], fc=CORE, ec="#555", lw=0.5)
    if show_network:
        g = to_networkx(c.topology)
        P = {k: np.asarray(v) for k, v in c.fit.positions.items()}
        for u, v in g.edges:
            if u in P and v in P:
                main = c.plan.edge_class.get((u, v), c.plan.edge_class.get((v, u))) == "main"
                ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color="#D9480F" if main else "#e8a37a", lw=1.4 if main else 0.8, zorder=4, alpha=0.9 if show_plan else 1.0)
        xs = [P[v][0] for v in g.nodes if v in P]; ys = [P[v][1] for v in g.nodes if v in P]
        roles = c.fit.roles
        ax.scatter(xs, ys, s=14, c=["#2B2B2B" if roles.get(v) in ("outer", "core") else "#1f77b4" for v in g.nodes if v in P], zorder=5, edgecolors="white", linewidths=0.5)
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, loc="left")


def candidate_to_png(c, path: str | Path, dpi: int = 160) -> Path:  # noqa: ANN001
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_candidate(ax, c, title=f"seed {c.seed} · {c.topology.num_nodes} 节点 · 走廊占比 {c.plan.diagnostics.get('corridor_ratio', 0) * 100:.0f}%")
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)
    return Path(path)


__all__ = ["candidate_to_geojson", "candidate_to_json", "candidate_to_png", "candidate_to_svg", "draw_candidate"]
