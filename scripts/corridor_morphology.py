#!/usr/bin/env python
"""Corridor-morphology statistics of real floor plans (Stage-3 target distribution).

Reads ``corridor_mask/{floor_id}_{region}.png`` (255 on corridors) and, when present, ``outer_mask`` for the
outline, and measures the quantities the Stage-3 renderer/fitter is calibrated against:

* corridor-area ratio (corridor px / outline px)
* corridor **width** distribution: 2 x distance-transform sampled along the medial axis (m); the split
  into *main* / *secondary* widths uses a 2-means split of the width samples
* **hierarchy**: share of medial-axis length that is main (wide) vs secondary
* **junction angles**: angles between skeleton branches at junction pixels (share < 45 deg / < 60 deg)
* **loops vs dead ends**: cycle rank of the skeleton graph, number of skeleton end-points, and how many of
  those end-points lie within ``--facade-reach`` widths of the facade (= entrances) vs inside (= vertical cores)
* **islands**: enclosed regions (holes) of the corridor body -> count and area distribution (the shop blocks)
* corridor centre-line distance from the facade for the outer loop (calibration of ``depth_area_coef``)

Pixel scale: ``--m-per-px`` (default from config ``stage3.m_per_px``) or, per floor, from the main table
(total_area / n_floors) when ``--corpus`` is given.

Usage (Mac):
  python scripts/corridor_morphology.py --config configs/data/legacy.yaml \
      --floors B0I0ZA8R1J_1_0 B0HUUZQJKM_1_0 ... --out outputs/experiments/corridor_morphology
  python scripts/corridor_morphology.py --config configs/data/legacy.yaml --floors-file docs/stage3_reference_floors.txt

Outputs ``per_floor.csv``, ``summary.json`` (medians / IQR of every metric), ``widths.csv`` (all width samples),
``angles.csv`` and a contact-sheet ``sheet.png`` (mask + skeleton coloured by width) for visual QA.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths  # noqa: E402
from mall_space_planner.utils.config import resolve_config  # noqa: E402


# ----------------------------------------------------------------------------------------------- image helpers
def _load_mask(path: Path, majority_ok: bool = False) -> np.ndarray:
    """255 = foreground. Corridor masks are sparse, so a mostly-white image means the convention is flipped;
    outline masks may legitimately be mostly white (``majority_ok``) — there the border decides."""
    from PIL import Image

    im = np.asarray(Image.open(path).convert("L"))
    fg = im > 127
    if majority_ok:
        border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
        if border.mean() > 0.5:  # foreground touches the whole frame -> inverted
            fg = ~fg
    elif fg.mean() > 0.5:
        fg = ~fg
    return fg


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    from skimage.measure import label

    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def _outline_from_corridor(corr: np.ndarray, close_px: int) -> np.ndarray:
    """Fallback outline when no outer_mask exists: closed + hole-filled corridor body."""
    from scipy import ndimage as ndi
    from skimage.morphology import binary_closing, disk

    body = binary_closing(corr, disk(close_px))
    return ndi.binary_fill_holes(body)


def _skeleton_graph(skel: np.ndarray):
    """8-connected pixel graph of the skeleton -> (networkx graph, coords dict)."""
    import networkx as nx

    ys, xs = np.nonzero(skel)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs, strict=True))}
    g = nx.Graph()
    g.add_nodes_from(range(len(ys)))
    for (y, x), i in idx.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy, dx) == (0, 0):
                    continue
                j = idx.get((y + dy, x + dx))
                if j is not None and j > i:
                    g.add_edge(i, j)
    pos = {i: np.array([x, y], float) for (y, x), i in idx.items()}
    return g, pos


def _junction_angles(g, pos, dist: np.ndarray, reach_px: int = 6) -> list[float]:
    """At each junction pixel (deg>=3), take the skeleton direction of each branch ``reach_px`` away and
    report pairwise angles. Junction clusters (adjacent junction pixels) are merged by taking the
    highest-degree pixel of each cluster."""
    import networkx as nx

    junc = [v for v in g.nodes if g.degree(v) >= 3]
    if not junc:
        return []
    sub = g.subgraph(junc)
    reps = [max(c, key=lambda v: (g.degree(v), -v)) for c in nx.connected_components(sub)]
    out: list[float] = []
    for v in reps:
        cluster = nx.node_connected_component(sub, v)
        dirs = []
        # walk each branch leaving the cluster
        frontier = {w for c in cluster for w in g.neighbors(c) if w not in cluster}
        for w in frontier:
            path = [w]
            prev = set(cluster)
            cur = w
            for _ in range(reach_px):
                nxt = [q for q in g.neighbors(cur) if q not in prev and q not in path]
                if len(nxt) != 1:
                    break
                prev.add(cur)
                cur = nxt[0]
                path.append(cur)
            d = pos[path[-1]] - pos[v]
            if np.linalg.norm(d) > 1e-9:
                dirs.append(d / np.linalg.norm(d))
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                out.append(float(np.degrees(np.arccos(np.clip(np.dot(dirs[i], dirs[j]), -1, 1)))))
    return out


def _prune_spurs(g, pos, dist: np.ndarray, rounds: int = 3):
    """Remove end branches shorter than the local corridor width (mask jaggies / raster spurs)."""
    g = g.copy()
    for _ in range(max(rounds, 1)):
        rm: set[int] = set()
        for v in [v for v in g.nodes if g.degree(v) == 1]:
            w = 2.0 * dist[int(pos[v][1]), int(pos[v][0])]
            chain, prev, cur = [v], None, v
            while True:
                nb = [q for q in g.neighbors(cur) if q != prev]
                if len(nb) != 1 or len(chain) > 4 * w + 8:
                    break
                prev, cur = cur, nb[0]
                if g.degree(cur) >= 3:
                    break
                chain.append(cur)
            if g.degree(cur) >= 3 and len(chain) <= max(w, 3.0):
                rm.update(chain)
        if not rm:
            break
        g.remove_nodes_from(rm)
        # collapse the pixels left with degree 0
        g.remove_nodes_from([v for v in g.nodes if g.degree(v) == 0 and g.number_of_nodes() > 1])
    return g


def _skeleton_loops(skel: np.ndarray, min_area_px: float) -> int:
    """Cycle rank of the corridor network = number of enclosed regions of the skeleton image
    (4-connected background components not touching the border) larger than ``min_area_px``.
    Robust to the spurious triangles an 8-connected pixel graph produces at diagonal steps."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(~skel, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return 0
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    areas = np.bincount(lab.ravel())
    return int(sum(1 for k in range(1, n + 1) if k not in border and areas[k] >= min_area_px))


def _two_means(x: np.ndarray) -> float:
    """1-D 2-means threshold (Otsu-like) — split of corridor widths into main / secondary."""
    x = np.sort(x)
    if len(x) < 4:
        return float(np.median(x)) if len(x) else 0.0
    best, thr = np.inf, float(np.median(x))
    for t in np.quantile(x, np.linspace(0.2, 0.8, 25)):
        a, b = x[x <= t], x[x > t]
        if len(a) == 0 or len(b) == 0:
            continue
        s = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if s < best:
            best, thr = s, float(t)
    return thr


# ----------------------------------------------------------------------------------------------- per floor
def analyse_floor(corr_path: Path, outline_path: Path | None, m_per_px: float, close_px: int, facade_reach: float, prune_px: int) -> tuple[dict, dict]:
    import networkx as nx
    from scipy import ndimage as ndi
    from skimage.morphology import medial_axis, remove_small_objects

    corr = _largest_blob(_load_mask(corr_path))
    corr = remove_small_objects(corr, 64)
    outline = _largest_blob(_load_mask(outline_path, majority_ok=True)) if outline_path else _outline_from_corridor(corr, close_px)
    if outline.shape != corr.shape:
        raise ValueError(f"shape mismatch corridor {corr.shape} vs outline {outline.shape}")
    area_out = float(outline.sum())
    area_corr = float(corr.sum())
    # skeleton (1-px, 8-connected) + local half-width from the medial-axis distance transform
    from skimage.morphology import skeletonize

    _, dist = medial_axis(corr, return_distance=True)
    dist = ndi.distance_transform_edt(corr)
    skel = skeletonize(corr)
    g, pos = _skeleton_graph(skel)
    g = _prune_spurs(g, pos, dist, rounds=prune_px)
    if g.number_of_nodes() == 0:
        raise ValueError("empty skeleton")
    keep = np.zeros_like(skel)
    for v in g.nodes:
        keep[int(pos[v][1]), int(pos[v][0])] = True
    skel = keep
    width_px = 2.0 * dist[skel]
    widths_m = width_px * m_per_px
    thr = _two_means(widths_m)
    main = widths_m > thr
    # loops / dead ends
    comps = list(nx.connected_components(g))
    big = g.subgraph(max(comps, key=len))
    ends = [v for v in big.nodes if big.degree(v) == 1]
    w_all_px = float(np.median(width_px)) if len(width_px) else 4.0
    cycle_rank = _skeleton_loops(skel, min_area_px=0.5 * w_all_px**2)
    # facade distance of every skeleton pixel
    d_out = ndi.distance_transform_edt(outline) * m_per_px
    end_fd = np.array([d_out[int(pos[v][1]), int(pos[v][0])] for v in ends]) if ends else np.zeros(0)
    w_main_m = float(np.median(widths_m[main])) if main.any() else float(np.median(widths_m))
    n_entr = int((end_fd <= facade_reach * w_main_m).sum())
    n_core = int(len(ends) - n_entr)
    # islands (holes of the closed corridor body inside the outline)
    body = ndi.binary_fill_holes(corr)
    holes = body & ~corr
    lab, n_holes = ndi.label(holes)
    hole_areas = np.bincount(lab.ravel())[1:] * m_per_px**2 if n_holes else np.zeros(0)
    hole_areas = hole_areas[hole_areas > 4.0]  # ignore pixel-noise holes < 4 m²
    # outer-loop facade distance: skeleton pixels on the outer boundary of the corridor body
    # ~ the 25% of skeleton pixels closest to the facade
    # outer loop = skeleton pixels whose corridor touches the outer shop band (outline minus filled corridor body).
    band = outline & ~body
    sk_yx = np.argwhere(skel)
    if band.any():
        d_band = ndi.distance_transform_edt(~band)
        on_outer = d_band[sk_yx[:, 0], sk_yx[:, 1]] <= dist[sk_yx[:, 0], sk_yx[:, 1]] + 1.5
    else:
        on_outer = np.ones(len(sk_yx), bool)
    fd_outer = d_out[sk_yx[on_outer, 0], sk_yx[on_outer, 1]]
    fd_outer = fd_outer[fd_outer > 0.75 * w_main_m]  # drop entrance stubs that run to the facade
    outer_fd = float(np.median(fd_outer)) if len(fd_outer) else float(np.median(d_out[skel]))
    angles = _junction_angles(g, pos, dist)
    A = np.asarray(angles, float)
    row = {
        "corridor_ratio": area_corr / area_out if area_out else np.nan,
        "area_outline_m2": area_out * m_per_px**2,
        "sqrt_area_m": float(np.sqrt(area_out)) * m_per_px,
        "width_med_m": float(np.median(widths_m)),
        "width_p10_m": float(np.quantile(widths_m, 0.1)),
        "width_p90_m": float(np.quantile(widths_m, 0.9)),
        "width_main_m": w_main_m,
        "width_sec_m": float(np.median(widths_m[~main])) if (~main).any() else np.nan,
        "main_len_share": float(main.mean()),
        "width_split_m": thr,
        "skel_len_m": float(skel.sum()) * m_per_px,
        "n_junctions": int(sum(1 for v in big.nodes if big.degree(v) >= 3)),
        "cycle_rank": int(cycle_rank),
        "n_dead_ends": int(len(ends)),
        "n_entrances_est": n_entr,
        "n_vertical_cores_est": n_core,
        "n_islands": int(len(hole_areas)),
        "island_area_med_m2": float(np.median(hole_areas)) if len(hole_areas) else np.nan,
        "island_area_max_m2": float(hole_areas.max()) if len(hole_areas) else np.nan,
        "outer_facade_dist_m": outer_fd,
        "outer_facade_over_sqrt_area": outer_fd / (float(np.sqrt(area_out)) * m_per_px) if area_out else np.nan,
        "angle_lt45": float(np.mean(A < 45)) if len(A) else np.nan,
        "angle_lt60": float(np.mean(A < 60)) if len(A) else np.nan,
        "angle_ortho_dev_med": float(np.median(np.abs(((A + 45) % 90) - 45))) if len(A) else np.nan,
        "n_angles": int(len(A)),
        "m_per_px": m_per_px,
    }
    extra = {"widths_m": widths_m, "angles": A, "skel": skel, "corr": corr, "outline": outline, "main": main}
    return row, extra


def _sheet(items: list[tuple[str, dict]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (fid, ex) in zip(axes, items, strict=False):
        ax.imshow(ex["outline"], cmap="gray_r", alpha=0.15)
        ax.imshow(np.ma.masked_where(~ex["corr"], ex["corr"]), cmap="Oranges", alpha=0.45, vmin=0, vmax=2)
        ys, xs = np.nonzero(ex["skel"])
        ax.scatter(xs, ys, s=0.4, c=np.where(ex["main"], "#b30000", "#1f77b4"), marker=".")
        ax.set_title(fid, fontsize=8)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data/legacy.yaml")
    p.add_argument("--floors", nargs="*", default=[], help="floor ids like B0I0ZA8R1J_1_0 (mall_floor_region)")
    p.add_argument("--floors-file", default=None, help="text file, one floor id per line (# comments allowed)")
    p.add_argument("--corridor-dir", default=None, help="override stage3.corridor_mask_dir")
    p.add_argument("--outline-dir", default=None, help="override stage3.outline_mask_dir (optional)")
    p.add_argument("--m-per-px", type=float, default=None)
    p.add_argument("--facade-reach", type=float, default=4.0, help="dead end within this many main-widths of the facade = entrance")
    p.add_argument("--prune-px", type=int, default=3)
    p.add_argument("--out", default="outputs/experiments/corridor_morphology")
    a = p.parse_args()
    cfg = resolve_config(a.config, [])
    s3 = Stage3Paths.from_config(cfg)
    ds = Stage3Dataset(s3)
    corr_dir = Path(a.corridor_dir).expanduser() if a.corridor_dir else s3.corridor_mask_dir
    out_dir = Path(a.outline_dir).expanduser() if a.outline_dir else s3.outline_mask_dir
    if corr_dir is None or not corr_dir.exists():
        raise SystemExit(f"corridor mask dir not found: {corr_dir}")
    floors = list(a.floors)
    if a.floors_file:
        floors += [ln.split("#")[0].strip() for ln in Path(a.floors_file).read_text().splitlines() if ln.split("#")[0].strip()]
    if not floors:
        floors = sorted(p_.stem for p_ in corr_dir.glob("*.png"))
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, sheet_items, all_w, all_a = [], [], [], []
    for fid in floors:
        parts = fid.split("_")
        floor_id, region = ("_".join(parts[:2]), int(parts[2])) if len(parts) >= 3 else (fid, 0)
        cp = corr_dir / f"{floor_id}_{region}.png"
        if not cp.exists():
            cp = corr_dir / f"{fid}.png"
        if not cp.exists():
            print(f"[skip] {fid}: no corridor mask at {cp}")
            rows.append({"floor_id": fid, "status": "missing"})
            continue
        op = (out_dir / cp.name) if out_dir and (out_dir / cp.name).exists() else None
        mpp = a.m_per_px or ds.m_per_px_for(floor_id, region)[0]
        try:
            row, ex = analyse_floor(cp, op, mpp, int(s3.outline_close_px), a.facade_reach, a.prune_px)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {fid}: {exc}")
            rows.append({"floor_id": fid, "status": f"error: {exc}"})
            continue
        row.update({"floor_id": fid, "status": "ok", "outline_source": "outer_mask" if op else "closed_corridor"})
        rows.append(row)
        sheet_items.append((fid, ex))
        all_w.append(pd.DataFrame({"floor_id": fid, "width_m": ex["widths_m"], "main": ex["main"]}))
        all_a.append(pd.DataFrame({"floor_id": fid, "angle_deg": ex["angles"]}))
        print(f"[ok] {fid}: ratio {row['corridor_ratio']:.3f} w_main {row['width_main_m']:.1f} m w_sec {row['width_sec_m']:.1f} m loops {row['cycle_rank']} dead-ends {row['n_dead_ends']} (entr {row['n_entrances_est']}, cores {row['n_vertical_cores_est']}) islands {row['n_islands']} <60deg {row['angle_lt60']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_floor.csv", index=False)
    ok = df[df["status"] == "ok"] if "status" in df else df
    num = ok.select_dtypes("number")
    summary = {
        "n_floors": int(len(ok)),
        "floors": list(ok["floor_id"]) if len(ok) else [],
        "median": {k: float(v) for k, v in num.median().items()},
        "q25": {k: float(v) for k, v in num.quantile(0.25).items()},
        "q75": {k: float(v) for k, v in num.quantile(0.75).items()},
    }
    if all_w:
        W = pd.concat(all_w)
        W.to_csv(out / "widths.csv", index=False)
        summary["width_pooled_m"] = {"p10": float(W.width_m.quantile(0.1)), "median": float(W.width_m.median()), "p90": float(W.width_m.quantile(0.9)), "main_median": float(W[W.main].width_m.median()) if W.main.any() else None, "sec_median": float(W[~W.main].width_m.median()) if (~W.main).any() else None}
    if all_a:
        Ang = pd.concat(all_a)
        Ang.to_csv(out / "angles.csv", index=False)
        summary["angles_pooled"] = {"lt45": float((Ang.angle_deg < 45).mean()), "lt60": float((Ang.angle_deg < 60).mean()), "n": int(len(Ang))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if sheet_items:
        _sheet(sheet_items, out / "sheet.png")
    print(json.dumps({k: summary[k] for k in summary if k != "floors"}, indent=1)[:3000])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
