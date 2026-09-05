"""Export of generated layouts to JSON / GeoJSON / SVG (PNG via matplotlib in visualization)."""

from __future__ import annotations

import json
from pathlib import Path

from mall_space_planner.schemas import GeneratedLayout


def layout_to_json(layout: GeneratedLayout, path: str | Path | None = None) -> str:
    s = layout.model_dump_json(indent=2)
    if path:
        Path(path).write_text(s, encoding="utf-8")
    return s


def layout_to_geojson(layout: GeneratedLayout, path: str | Path | None = None) -> dict:
    feats = [{"type": "Feature", "properties": {"kind": "boundary"}, "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in layout.boundary.exterior + layout.boundary.exterior[:1]]]}}]
    for n, (x, y) in layout.skeleton_positions.items():
        feats.append({"type": "Feature", "properties": {"kind": "junction", "id": n}, "geometry": {"type": "Point", "coordinates": [x, y]}})
    for u, v in layout.topology.edges():
        if u in layout.skeleton_positions and v in layout.skeleton_positions:
            feats.append({"type": "Feature", "properties": {"kind": "corridor", "u": u, "v": v}, "geometry": {"type": "LineString", "coordinates": [list(layout.skeleton_positions[u]), list(layout.skeleton_positions[v])]}})
    for unit in layout.units:
        if unit.polygon:
            feats.append({"type": "Feature", "properties": {"kind": unit.kind, "id": unit.unit_id, "area": unit.area}, "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in unit.polygon + unit.polygon[:1]]]}})
    fc = {"type": "FeatureCollection", "features": feats}
    if path:
        Path(path).write_text(json.dumps(fc, ensure_ascii=False, indent=1), encoding="utf-8")
    return fc


def layout_to_svg(layout: GeneratedLayout, path: str | Path | None = None, size: int = 800) -> str:
    xs = [p[0] for p in layout.boundary.exterior]
    ys = [p[1] for p in layout.boundary.exterior]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    s = size * 0.9 / span

    def t(p: tuple[float, float]) -> tuple[float, float]:
        return (size * 0.05 + (p[0] - minx) * s, size - (size * 0.05 + (p[1] - miny) * s))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">']
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(t, layout.boundary.exterior))
    parts.append(f'<polygon points="{pts}" fill="#f7f7f7" stroke="#333" stroke-width="2"/>')
    for unit in layout.units:
        if unit.polygon:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(t, unit.polygon))
            parts.append(f'<polygon points="{pts}" fill="#cfe8ff" stroke="#3a7bd5" stroke-width="1"/>')
    for u, v in layout.topology.edges():
        if u in layout.skeleton_positions and v in layout.skeleton_positions:
            (x1, y1), (x2, y2) = t(layout.skeleton_positions[u]), t(layout.skeleton_positions[v])
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d9480f" stroke-width="3"/>')
    for n, p in layout.skeleton_positions.items():
        x, y = t(p)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#222"/><text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="10">{n}</text>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    if path:
        Path(path).write_text(svg, encoding="utf-8")
    return svg
