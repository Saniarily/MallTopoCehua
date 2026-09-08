"""Matplotlib figures shared by the CLI scripts, the FastAPI backend and the Streamlit hub.

* :func:`draw_renovation` – the four-panel renovation figure (real network | regrown network | corridor plan | metric
  table) used by ``scripts/renovate_stage3.py`` and by the hub.
* :func:`draw_network_in_outline` – one network drawn at given positions inside an outline (skeleton edges emphasised).
* :func:`draw_case_topology` – abstract topology drawing of a database case (skeleton highlighted), via reporting.graphdraw.
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from mall_space_planner.stage3.renovate import BETTER, TOPO_LABEL  # noqa: E402
from mall_space_planner.topology.convert import to_networkx  # noqa: E402


_STYLED = False


def _style() -> None:
    """Apply the thesis matplotlib style (CJK-safe font chain) once per process. Called at import so every figure drawn
    by the hub (thumbnails, plates, candidates, Streamlit pages) renders Chinese labels; DejaVu Sans stays in the chain
    for glyphs the CJK font lacks (e.g. the superscript ² in m²)."""
    global _STYLED
    if _STYLED:
        return
    try:
        from mall_space_planner.reporting.style import apply_style

        apply_style()
        import warnings

        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)
        _STYLED = True
    except Exception:  # noqa: BLE001
        pass


ensure_style = _style
_style()


def poly(ax, geom, **kw) -> None:  # noqa: ANN001
    """Draw a (Multi)Polygon with holes."""
    for p in getattr(geom, "geoms", [geom]):
        if p is None or p.is_empty:
            continue
        ax.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, **kw))
        for ring in p.interiors:
            ax.add_patch(MplPolygon(np.array(ring.coords), closed=True, fc="white", ec=kw.get("ec", "none"), lw=kw.get("lw", 0.5)))


NETWORK_STYLES = {
    # orange: generated / renovation figures (network is the subject, background is flat grey)
    "orange": {"main": "#D9480F", "sec": "#e8a37a", "lw_main": 1.5, "lw_sec": 0.9, "node_sk": "#2B2B2B", "node": "#1f77b4", "alpha": 1.0},
    # ink: real floor plates over a colour-block plan – 90 % black lines read clearly on pastel blocks
    "ink": {"main": "#000000", "sec": "#000000", "lw_main": 1.6, "lw_sec": 0.7, "node_sk": "#000000", "node": "#444444", "alpha": 0.9},
}


def draw_network_in_outline(ax, topo, pos: dict, outline, skeleton_nodes: set | None = None, title: str = "", anchored=None, node_size: float = 16.0, style: str = "orange") -> None:  # noqa: ANN001
    st = NETWORK_STYLES[style]
    sk = set(skeleton_nodes or ())
    g = to_networkx(topo)
    if outline is not None:
        poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
    for u, v in g.edges:
        if u in pos and v in pos:
            main = u in sk and v in sk
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=st["main"] if main else st["sec"], lw=st["lw_main"] if main else st["lw_sec"], alpha=st["alpha"], zorder=3, solid_capstyle="round")
    nodes = [v for v in g.nodes if v in pos]
    ax.scatter([pos[v][0] for v in nodes], [pos[v][1] for v in nodes], s=node_size, c=[st["node_sk"] if v in sk else st["node"] for v in nodes], alpha=st["alpha"], zorder=5, edgecolors="white", linewidths=0.5)
    for v in anchored or ():
        if v in pos:
            ax.scatter([pos[v][0]], [pos[v][1]], s=70, facecolors="none", edgecolors="#D9480F", linewidths=1.2, zorder=6)
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def draw_plan(ax, plan, outline, title: str = "") -> None:  # noqa: ANN001
    poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
    poly(ax, plan.corridors_secondary, fc="#F7DDB0", ec="#b07a2a", lw=0.4)
    poly(ax, plan.corridors_main, fc="#F0C987", ec="#b07a2a", lw=0.5)
    for at in plan.atria:
        poly(ax, at, fc="#B5E7A0", ec="#5a9a4a", lw=0.5)
    for e in plan.entrances:
        poly(ax, e["stub"], fc="#F0C987", ec="#b07a2a", lw=0.4)
        ax.scatter([e["point"][0]], [e["point"][1]], marker="v", s=60, c="#D9480F", zorder=6)
    for vc in plan.vertical_cores:
        poly(ax, vc["polygon"], fc="#9e9e9e", ec="#555", lw=0.5)
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def indicator_rows(ind_b: dict, ind_a: dict, row: dict | None = None) -> list[list[str]]:
    keys = ["pred_score", "num_cycles", "avg_shortest_path", "diameter", "closeness_mean", "max_betweenness", "degree_entropy", "avg_degree", "n_dead_ends"]
    rows = []
    for k in keys:
        b, a = ind_b.get(k), ind_a.get(k)
        if b is None or a is None:
            continue
        better = BETTER.get(k, 0)
        arrow = "" if better == 0 or abs(a - b) < 1e-9 else ("▲" if (a - b) * better > 0 else "▼")
        rows.append([TOPO_LABEL[k], f"{b:.2f}" if isinstance(b, float) else str(b), f"{a:.2f}" if isinstance(a, float) else str(a), arrow])
    if row and row.get("before_sharp_angle_rate") is not None:
        b, a = row["before_sharp_angle_rate"], row["after_sharp_angle_rate"]
        rows.append(["锐角(<60°)比例", f"{b:.2f}", f"{a:.2f}", "" if abs(a - b) < 1e-9 else ("▲" if a < b else "▼")])
    return rows


def draw_renovation(r: dict[str, Any], gen_name: str | None = None, title: str | None = None):  # noqa: ANN201
    """Four-panel renovation figure from a ``renovate_floor`` result dict. Returns the Figure (caller saves/closes)."""
    _style()
    outline, sk = r["outline"], r["sk_nodes"]
    gen_name = gen_name or r.get("generator_name", "")
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    ib, ia = r["ind_b"], r["ind_a"]
    draw_network_in_outline(axes[0], r["before"], r["gt"], outline, sk, f"① 现状：真实关键点网络\n{ib['num_nodes']} 节点 · {ib['num_cycles']} 回路 · ASPL {ib['avg_shortest_path']:.2f}")
    P = {k: np.asarray(v) for k, v in r["res"].positions.items()}
    draw_network_in_outline(axes[1], r["after"], P, outline, sk, f"② 更新：{gen_name}（黑 = 原型节点，位置重排；红框 = 保留的出入口）\n{ia['num_nodes']} 节点 · {ia['num_cycles']} 回路 · ASPL {ia['avg_shortest_path']:.2f}", anchored=r.get("anchored", ()))
    d = r["plan"].diagnostics
    draw_plan(axes[2], r["plan"], outline, f"③ 走廊布局方案\n主廊 {d['main_width_m']:.0f} m / 次廊 {d['secondary_width_m']:.0f} m · {d['n_entrances']} 出入口 · {d['n_vertical_cores']} 竖向核 · {d['n_atria']} 中庭 · 占比 {d['corridor_ratio'] * 100:.0f}%")
    ax = axes[3]; ax.axis("off")
    rows = indicator_rows(ib, ia, r.get("row"))
    if rows:
        tbl = ax.table(cellText=rows, colLabels=["指标", "现状", "更新", ""], loc="center", cellLoc="center", colWidths=[0.5, 0.18, 0.18, 0.1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.35)
        for (i, j), c in tbl.get_celld().items():
            c.set_edgecolor("#bbb")
            if i == 0:
                c.set_text_props(fontweight="bold")
            if j == 3 and i > 0:
                c.set_text_props(color="#2e7d32" if rows[i - 1][3] == "▲" else ("#c62828" if rows[i - 1][3] == "▼" else "#333"))
    ax.set_title("④ 关键拓扑指标：现状 vs 更新（▲ 改善）", fontsize=9)
    if title is None:
        title = f"旧商场改造：{r.get('floor_id', '')}（同一轮廓、保留出入口位置、重新生长网络并重排节点）"
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def draw_case_topology(ax, topo, skeleton_nodes: set | None = None, title: str = "") -> None:  # noqa: ANN001
    """Abstract (position-free) topology drawing of a database case."""
    from mall_space_planner.reporting.graphdraw import draw_topology

    _style()
    draw_topology(ax, topo, skeleton_nodes, title=title)
