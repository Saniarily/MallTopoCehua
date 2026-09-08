"""Floor-plate figures of the real data: colour-block plan + M network, outline + M network, square thumbnails.

Used by ``scripts/export_floor_plates.py`` (thesis appendix folders, Viewer Hub thumbnail set) and by
``Workbench.floor_thumbnail`` (which prefers a pre-rendered file from ``outputs/floor_plates/thumbs``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mall_space_planner.hub.viz import draw_network_in_outline, poly  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLATES_DIR = ROOT / "outputs" / "floor_plates"
THUMB_PX = 360


def floor_ids_from_graph_dir(gd: Path) -> list[str]:
    return sorted({p.name[: -len("_M.csv")] for p in gd.glob("*_M.csv") if not p.name.endswith("_M_simplified.csv") and (gd / f"{p.name[: -len('_M.csv')]}_total.csv").exists()})


def load_floor(fid: str, ds, gd: Path | None = None) -> dict[str, Any]:  # noqa: ANN001
    """Full network, skeleton, aligned real positions, outline, plan PNG path for one floor."""
    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.data.legacy_adapter import load_graph_csv
    from mall_space_planner.stage3.outline import m_positions_from_total_csv

    gd = gd or ds.paths.graph_dir
    full = load_target_csv(gd / f"{fid}_M.csv")
    skp = gd / f"{fid}_M_simplified.csv"
    sk = load_graph_csv(skp, gd / f"{fid}_M_simplified_node_attributes.csv") if skp.exists() else None
    outline = ds.outline(fid, 0)
    gt = m_positions_from_total_csv(gd / f"{fid}_total.csv", outline)
    return {"full": full, "skeleton": sk, "positions": gt, "outline": outline, "plan_png": ds.paths.plan_png(fid, 0), "graph_dir": gd}


def _plan_background(ax, outline, plan_png: Path | None, alpha: float = 0.95) -> bool:  # noqa: ANN001
    if plan_png is None or not Path(plan_png).exists():
        return False
    try:
        from PIL import Image

        img = np.asarray(Image.open(plan_png).convert("RGB"))
        H, W = img.shape[:2]
        (x0, y0), (x1, y1) = outline.px_to_m(np.array([[0.0, 0.0], [float(W), float(H)]]))
        ax.imshow(img, extent=(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)), interpolation="bilinear", zorder=0, alpha=alpha)
        return True
    except Exception:  # noqa: BLE001
        return False


def draw_plate(ax, rn: dict[str, Any], with_plan: bool, title: str = "", node_size: float = 14.0, square: bool = False) -> bool:  # noqa: ANN001
    """One floor plate on ``ax``. Returns whether the colour plan was used as background."""
    outline = rn["outline"]
    used = _plan_background(ax, outline, rn.get("plan_png")) if with_plan else False
    if not used:
        poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=0.9)
    sk = set(rn["skeleton"].nodes) if rn.get("skeleton") is not None else set()
    draw_network_in_outline(ax, rn["full"], rn["positions"], None, sk, node_size=node_size)
    poly(ax, outline.polygon, fc="none", ec="#222", lw=1.0)
    minx, miny, maxx, maxy = outline.polygon.bounds
    if square:
        cx, cy, half = (minx + maxx) / 2, (miny + maxy) / 2, 0.53 * max(maxx - minx, maxy - miny)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    else:
        mx, my = 0.03 * (maxx - minx), 0.03 * (maxy - miny)
        ax.set_xlim(minx - mx, maxx + mx); ax.set_ylim(miny - my, maxy + my)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return used


def render_thumbnail_bytes(rn: dict[str, Any], size_px: int = THUMB_PX) -> bytes:
    import io

    dpi = 100
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    draw_plate(ax, rn, with_plan=True, node_size=9, square=True)
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, facecolor="white"); plt.close(fig)
    return buf.getvalue()


def render_floor_plates(fid: str, ds, out: Path, cases=None, force: bool = False, thumb_px: int = THUMB_PX, dpi: int = 130) -> dict[str, Any]:  # noqa: ANN001
    """Write plan_topo / outline_topo / thumbs PNGs for one floor and return its metadata row."""
    from shapely.geometry import Point

    from mall_space_planner.data.legacy_adapter import split_floor_id
    from mall_space_planner.topology.convert import to_networkx

    out = Path(out)
    targets = {k: out / k / f"{fid}.png" for k in ("plan_topo", "outline_topo", "thumbs")}
    mall, k = split_floor_id(fid)
    row: dict[str, Any] = {"floor_id": fid, "mall_id": mall, "floor": k}
    if cases is not None and len(cases):
        idc = "floor_id" if "floor_id" in cases.columns else cases.columns[0]
        c = cases[cases[idc].astype(str) == fid]
        if not c.empty:
            c = c.iloc[0]
            row.update({"score": c.get("total_score"), "layout_type": c.get("layout_type"), "split": c.get("split")})
    rn = load_floor(fid, ds)
    outline, gt = rn["outline"], rn["positions"]
    g = to_networkx(rn["full"])
    al = outline.extra.get("csv_align") or {}
    row.update({"n_nodes": rn["full"].num_nodes, "n_edges": rn["full"].num_edges, "n_skeleton": rn["skeleton"].num_nodes if rn["skeleton"] is not None else None, "area_m2": round(outline.area),
                "n_entrances": int(sum(1 for v in g.nodes if g.degree(v) == 1 and v in gt and outline.polygon.exterior.distance(Point(gt[v])) <= 40.0)),
                "nodes_outside": int(sum(1 for v in g.nodes if v in gt and not outline.polygon.buffer(1.0).covers(Point(gt[v])))),
                "align_mode": al.get("mode"), "align_shift_px": json.dumps(al.get("shift_px")) if al else None, "align_scale": json.dumps(al.get("scale")) if al else None,
                "has_plan_png": bool(rn["plan_png"] is not None and Path(rn["plan_png"]).exists())})
    if not force and all(t.exists() for t in targets.values()):
        row["status"] = "exists"
        return row
    for t in targets.values():
        t.parent.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = outline.polygon.bounds
    w, h = maxx - minx, maxy - miny
    fw = 7.0; fh = max(3.0, min(9.0, fw * h / max(w, 1e-9) + 0.5))
    ttl = f"{fid} · {row['n_nodes']} 节点 / {row['n_edges']} 连接 · {row['area_m2']:,} m²" + (f" · 评分 {row['score']}" if row.get("score") is not None else "")
    fig, ax = plt.subplots(figsize=(fw, fh))
    used = draw_plate(ax, rn, with_plan=True, title=ttl)
    fig.savefig(targets["plan_topo"], dpi=dpi, bbox_inches="tight", facecolor="white"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(fw, fh))
    draw_plate(ax, rn, with_plan=False, title=ttl)
    fig.savefig(targets["outline_topo"], dpi=dpi, bbox_inches="tight", facecolor="white"); plt.close(fig)
    targets["thumbs"].write_bytes(render_thumbnail_bytes(rn, thumb_px))
    row.update({"status": "ok", "plan_bg_used": used})
    return row


def find_prerendered_thumbnail(fid: str, plates_dir: Path | None = None) -> Path | None:
    p = (plates_dir or DEFAULT_PLATES_DIR) / "thumbs" / f"{fid}.png"
    return p if p.exists() else None


def load_plates_table(plates_dir: Path | None = None):  # noqa: ANN201
    p = (plates_dir or DEFAULT_PLATES_DIR) / "floors.csv"
    if not p.exists():
        return None
    import pandas as pd

    return pd.read_csv(p)
