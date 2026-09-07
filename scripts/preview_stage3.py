#!/usr/bin/env python3
"""Stage-3 preview: fit an M network into an outline and render the corridor system.

  # real floor: outline + ground-truth M positions from *_total.csv, topology from *_M.csv
  python scripts/preview_stage3.py --total tests/fixtures/graph_csv/B000A0E928_1_total.csv --m tests/fixtures/graph_csv/B000A0E928_1_M.csv --out /tmp/s3.png
  # same topology into a different outline (another floor's total.csv or a hand-drawn polygon json)
  python scripts/preview_stage3.py --total A_total.csv --m A_M.csv --outline-total B_total.csv --out /tmp/s3_transfer.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MP  # noqa: E402

from mall_space_planner.data.corpus_builder import load_target_csv  # noqa: E402
from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, outline_from_points, outline_from_total_csv, render_corridors  # noqa: E402
from mall_space_planner.stage3.evaluate import evaluate_fit  # noqa: E402
from mall_space_planner.stage3.outline import corridor_polygons_from_total_csv, m_positions_from_total_csv  # noqa: E402
from mall_space_planner.topology.convert import to_networkx  # noqa: E402


def _draw_poly(ax, poly, **kw):  # noqa: ANN001
    for p in getattr(poly, "geoms", [poly]):
        if p.is_empty:
            continue
        ax.add_patch(MP(np.array(p.exterior.coords), closed=True, **kw))
        for r in p.interiors:
            ax.add_patch(MP(np.array(r.coords), closed=True, fc="white", ec=kw.get("ec", "none"), lw=kw.get("lw", 0.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", required=True)
    ap.add_argument("--m", required=True, help="*_M.csv (complete M network) or *_M_simplified.csv")
    ap.add_argument("--outline-total", default=None, help="use another floor's total.csv as the target outline")
    ap.add_argument("--outline-json", default=None, help="hand-drawn polygon: JSON list of [x, y] in metres")
    ap.add_argument("--area", type=float, default=None, help="gross floor area m2 (for pixel scale)")
    ap.add_argument("--shop-depth", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/stage3_preview.png")
    a = ap.parse_args()

    src_outline = outline_from_total_csv(a.total, area_m2=a.area)
    topo = load_target_csv(Path(a.m))
    gt_pos = m_positions_from_total_csv(a.total, src_outline)
    gt_pos = {k: v for k, v in gt_pos.items() if k in set(topo.nodes)}
    if a.outline_json:
        outline = outline_from_points(json.load(open(a.outline_json)))
        transfer = True
    elif a.outline_total:
        outline = outline_from_total_csv(a.outline_total, area_m2=a.area)
        transfer = True
    else:
        outline, transfer = src_outline, False

    fitter = CorridorFitter(FitParams(shop_depth=a.shop_depth))
    res = fitter.fit(topo, outline, seed=a.seed)
    plan = render_corridors(topo, res.positions, outline, res.roles, RenderParams())
    ev = evaluate_fit(topo, res, plan, outline, gt_positions=None if transfer else gt_pos)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev.items()}, ensure_ascii=False, indent=1))

    ncol = 3 if not transfer else 2
    f, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5.6))
    g = to_networkx(topo)
    # (a) real drawing
    if not transfer:
        ax = axes[0]
        _draw_poly(ax, src_outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
        for cp in corridor_polygons_from_total_csv(a.total, src_outline):
            _draw_poly(ax, cp, fc="#F0C987", ec="#b07a2a", lw=0.3)
        for u, v in g.edges:
            if u in gt_pos and v in gt_pos:
                ax.plot([gt_pos[u][0], gt_pos[v][0]], [gt_pos[u][1], gt_pos[v][1]], color="#D9480F", lw=0.8)
        ax.scatter(*np.array(list(gt_pos.values())).T, s=10, c="k", zorder=5)
        ax.set_title(f"real: {len(gt_pos)} M nodes, corridors from *_total.csv", fontsize=9)
    # (b) fit
    ax = axes[-2]
    _draw_poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
    _draw_poly(ax, res.inset, fc="none", ec="#999", lw=0.6)
    for l in res.axis_lines:
        ax.plot(*np.array(l.coords).T, color="#bbb", lw=0.6, ls="--")
    P = res.positions
    colors = {"outer": "#0072B2", "core": "#009E73", "branch": "#E69F00", "leaf": "#CC79A7"}
    for u, v in g.edges:
        ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color="#D9480F" if plan.edge_class.get((u, v), plan.edge_class.get((v, u))) == "main" else "#e8a37a", lw=1.2 if plan.edge_class.get((u, v), plan.edge_class.get((v, u))) == "main" else 0.8)
    ax.scatter([P[v][0] for v in g.nodes], [P[v][1] for v in g.nodes], s=16, c=[colors[res.roles[v]] for v in g.nodes], zorder=5, edgecolors="k", linewidths=0.4)
    ax.set_title(f"fit: score {res.score:.2f}, cross {ev['crossings']}, inside {ev['inside_ratio']:.2f}, ortho {ev['ortho_deviation_deg']:.0f}°" + (f", Procrustes {ev['procrustes_rmse_m']:.1f} m" if "procrustes_rmse_m" in ev else ""), fontsize=9)
    # (c) rendered corridors
    ax = axes[-1]
    _draw_poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
    _draw_poly(ax, plan.corridors_secondary, fc="#F7DDB0", ec="#b07a2a", lw=0.4)
    _draw_poly(ax, plan.corridors_main, fc="#F0C987", ec="#b07a2a", lw=0.5)
    for at in plan.atria:
        _draw_poly(ax, at, fc="#B5E7A0", ec="#5a9a4a", lw=0.5)
    for e in plan.entrances:
        _draw_poly(ax, e["stub"], fc="#F0C987", ec="#b07a2a", lw=0.4)
        ax.scatter([e["point"][0]], [e["point"][1]], marker="v", s=60, c="#D9480F", zorder=6)
    d = plan.diagnostics
    ax.set_title(f"corridors: main {d['main_length_m']:.0f} m / sec {d['secondary_length_m']:.0f} m, {d['n_entrances']} entrances, {d['n_atria']} atria, {d['corridor_ratio']*100:.0f}% of floor", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
    f.tight_layout()
    f.savefig(a.out, dpi=100)
    print(a.out)


if __name__ == "__main__":
    main()
