#!/usr/bin/env python
"""Stage 3 – *renovation* workflow: keep the existing mall's outline, main corridors and entrances, regrow the rest.

Most projects today are refurbishments of built (often low-rated, grid-like) malls rather than new builds. For a floor
``{mall}_{k}`` this script:

1. loads the real complete key-point network ``*_M.csv`` (before) with its real positions from ``*_total.csv`` and the
   Stage-1 skeleton ``*_M_simplified.csv`` (the corridors that stay – structure, main loops, entrances);
2. regrows a new complete network from that skeleton with the Stage-2 generator (AR-GNN best-of-16 when a checkpoint is
   available, else rule best-of-16) to the same node count (after);
3. fits the new network into the **same outline** with the skeleton nodes **anchored at their real positions**
   (``CorridorFitter.fit(anchors=...)``) and renders the corridor plan (main = skeleton corridors, entrances at the real
   dead ends, vertical cores at interior dead ends, atria);
4. compares before / after on topology indicators (cycle count, avg degree, ASPL, integration-like closeness, degree
   entropy, max betweenness, dead ends) and on the corridor plan (corridor ratio, sharp angles, entrances, atria).

Outputs per floor: ``<out>/<floor>.png`` (three panels: real network | regrown network | corridor plan, plus a metric
table) and rows in ``<out>/per_floor.csv``; ``<out>/summary.json`` aggregates the deltas.

Usage (Mac):
  python scripts/renovate_stage3.py --config configs/data/legacy.yaml --floors B000A0E928_1 B0HG3AE08N_1 --out outputs/experiments/renovation
  python scripts/renovate_stage3.py --config configs/data/legacy.yaml --split test --limit 40 --low-score 4.3 --out outputs/experiments/renovation
  (--low-score keeps only malls whose total_score <= threshold: the renovation candidates)
Sandbox smoke test (fixture floor, rule generator):
  python scripts/renovate_stage3.py --graph-dir tests/fixtures/graph_csv --floors B000A0E928_1 --out /tmp/renov
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mall_space_planner.data.corpus_builder import load_target_csv  # noqa: E402
from mall_space_planner.data.legacy_adapter import load_graph_csv, split_floor_id  # noqa: E402
from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, render_corridors  # noqa: E402
from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths  # noqa: E402
from mall_space_planner.stage3.renovate import BETTER, TOPO_LABEL, build_generator, renovate_floor  # noqa: E402
from mall_space_planner.topology.convert import to_networkx  # noqa: E402
from mall_space_planner.utils.config import resolve_config  # noqa: E402

def draw(r: dict, out_png: Path, gen_name: str, title: str) -> None:  # noqa: ANN001
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    from mall_space_planner.reporting.style import apply_style

    try:
        apply_style()
    except Exception:  # noqa: BLE001
        pass
    outline, sk = r["outline"], r["sk_nodes"]

    def poly(ax, geom, **kw):  # noqa: ANN001, ANN202
        for p in getattr(geom, "geoms", [geom]):
            if p.is_empty:
                continue
            ax.add_patch(MplPolygon(np.array(p.exterior.coords), closed=True, **kw))
            for ring in p.interiors:
                ax.add_patch(MplPolygon(np.array(ring.coords), closed=True, fc="white", ec=kw.get("ec", "none"), lw=kw.get("lw", 0.5)))

    def net(ax, topo, pos, sk_nodes, ttl):  # noqa: ANN001, ANN202
        g = to_networkx(topo)
        poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=1.0)
        for u, v in g.edges:
            if u in pos and v in pos:
                main = u in sk_nodes and v in sk_nodes
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#D9480F" if main else "#e8a37a", lw=1.5 if main else 0.9, zorder=3)
        xs = [pos[v][0] for v in g.nodes if v in pos]; ys = [pos[v][1] for v in g.nodes if v in pos]
        col = ["#2B2B2B" if v in sk_nodes else "#1f77b4" for v in g.nodes if v in pos]
        ax.scatter(xs, ys, s=16, c=col, zorder=5, edgecolors="white", linewidths=0.5)
        ax.set_aspect("equal"); ax.autoscale(); ax.axis("off"); ax.set_title(ttl, fontsize=9)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    ib, ia = r["ind_b"], r["ind_a"]
    net(axes[0], r["before"], r["gt"], sk, f"① 现状：真实关键点网络\n{ib['num_nodes']} 节点 · {ib['num_cycles']} 回路 · ASPL {ib['avg_shortest_path']:.2f}")
    P = {k: np.asarray(v) for k, v in r["res"].positions.items()}
    net(axes[1], r["after"], P, sk, f"② 更新：{gen_name}（黑 = 保留的主走廊节点）\n{ia['num_nodes']} 节点 · {ia['num_cycles']} 回路 · ASPL {ia['avg_shortest_path']:.2f}")
    ax = axes[2]
    plan = r["plan"]
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
    d = plan.diagnostics
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    ax.set_title(f"③ 走廊布局方案\n主廊 {d['main_width_m']:.0f} m / 次廊 {d['secondary_width_m']:.0f} m · {d['n_entrances']} 出入口 · {d['n_vertical_cores']} 竖向核 · {d['n_atria']} 中庭 · 占比 {d['corridor_ratio']*100:.0f}%", fontsize=9)
    # metric table
    ax = axes[3]; ax.axis("off")
    keys = ["num_cycles", "avg_shortest_path", "diameter", "closeness_mean", "max_betweenness", "degree_entropy", "avg_degree", "n_dead_ends"]
    rows = []
    for k in keys:
        b, a = ib[k], ia[k]
        if b is None or a is None:
            continue
        better = BETTER.get(k, 0)
        arrow = "" if better == 0 or abs(a - b) < 1e-9 else ("▲" if (a - b) * better > 0 else "▼")
        rows.append([TOPO_LABEL[k], f"{b:.2f}" if isinstance(b, float) else str(b), f"{a:.2f}" if isinstance(a, float) else str(a), arrow])
    row_r = r["row"]
    if row_r.get("before_sharp_angle_rate") is not None:
        b, a = row_r["before_sharp_angle_rate"], row_r["after_sharp_angle_rate"]
        rows.append(["锐角(<60°)比例", f"{b:.2f}", f"{a:.2f}", "" if abs(a - b) < 1e-9 else ("▲" if a < b else "▼")])
    tbl = ax.table(cellText=rows, colLabels=["指标", "现状", "更新", ""], loc="center", cellLoc="center", colWidths=[0.5, 0.18, 0.18, 0.1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.35)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#bbb")
        if i == 0:
            c.set_text_props(fontweight="bold")
        if j == 3 and i > 0:
            c.set_text_props(color="#2e7d32" if rows[i - 1][3] == "▲" else ("#c62828" if rows[i - 1][3] == "▼" else "#333"))
    ax.set_title("④ 关键拓扑指标：现状 vs 更新（▲ 改善）", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/legacy.yaml")
    ap.add_argument("--graph-dir", default=None, help="override dataset.params.graph_dir (folder with *_M.csv / *_M_simplified*.csv / *_total.csv)")
    ap.add_argument("--floors", nargs="*", default=[])
    ap.add_argument("--split", default=None, help="take floors from the Stage-2 corpus split instead of --floors")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--low-score", type=float, default=None, help="keep only malls with total_score <= this (needs data/processed/legacy/cases.csv)")
    ap.add_argument("--max-nodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=6, help="generator samples per floor; the one with the best renovation objective (fewest new dead ends, loops / ASPL not worse) is kept")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--results", default="data/results_snapshot")
    ap.add_argument("--out", default="outputs/experiments/renovation")
    a = ap.parse_args()
    cfg = resolve_config(a.config, [])
    s3 = Stage3Paths.from_config(cfg)
    if a.graph_dir:
        s3.graph_dir = Path(a.graph_dir).expanduser()
    if s3.graph_dir is None or not s3.graph_dir.exists():
        raise SystemExit(f"graph dir not found: {s3.graph_dir} (use --graph-dir)")
    ds = Stage3Dataset(s3)
    floors = list(a.floors)
    if a.split:
        from mall_space_planner.data.corpus_builder import load_corpus_jsonl

        corpus = Path(a.corpus or cfg.get("stage2_corpus", "data/processed/legacy/stage2_corpus_v2.jsonl"))
        smp = load_corpus_jsonl(corpus, split=a.split)
        floors += [s.sample_id for s in smp]
    if a.low_score is not None:
        cases = Path(cfg.get("data", {}).get("processed_dir", "data/processed/legacy")) / "cases.csv"
        if cases.exists():
            df = pd.read_csv(cases)
            low = set(df.loc[df["total_score"] <= a.low_score, "mall_id"].astype(str))
            floors = [f for f in floors if split_floor_id(f)[0] in low]
            print(f"low-score filter (<= {a.low_score}): {len(floors)} floors")
        else:
            print(f"[warn] {cases} not found; --low-score ignored")
    floors = floors[: a.limit] if a.limit else floors
    if not floors:
        raise SystemExit("no floors selected")
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen, gen_name = build_generator(Path(a.results))
    shop_depth = float((cfg.get("stage3") or {}).get("shop_depth_m", 14.0)); cr = float((cfg.get("stage3") or {}).get("corridor_ratio", 0.18))
    fitter = CorridorFitter(FitParams(shop_depth=shop_depth, n_restarts=a.restarts, iters=a.iters)); rp = RenderParams(corridor_ratio=cr)
    rows = []
    for i, fid in enumerate(floors):
        t0 = time.time()
        try:
            n_probe = load_target_csv(s3.graph_dir / f"{fid}_M.csv").num_nodes if (s3.graph_dir / f"{fid}_M.csv").exists() else None
            if n_probe is not None and n_probe > a.max_nodes:
                rows.append({"floor_id": fid, "status": "skipped_large", "n_nodes": n_probe}); continue
            r = renovate_floor(fid, s3.graph_dir, ds, gen, fitter, rp, seed=a.seed, n_candidates=a.candidates)
            draw(r, out / f"{fid}.png", gen_name, f"旧商场改造：{fid}（同一轮廓、保留主走廊与出入口，重新生长次级网络）")
            r["row"].update({"status": "ok", "generator": gen_name, "seconds": round(time.time() - t0, 1)})
            rows.append(r["row"])
            ib, ia = r["ind_b"], r["ind_a"]
            print(f"[{i + 1}/{len(floors)}] {fid}: cycles {ib['num_cycles']}->{ia['num_cycles']}  ASPL {ib['avg_shortest_path']:.2f}->{ia['avg_shortest_path']:.2f}  dead-ends {ib['n_dead_ends']}->{ia['n_dead_ends']}  cross {r['row']['after_crossings']}  entr {r['row']['after_n_entrances']}  atria {r['row']['after_n_atria']}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback

            tb = traceback.extract_tb(exc.__traceback__)[-1]
            rows.append({"floor_id": fid, "status": f"error: {type(exc).__name__}: {exc}", "error_at": f"{Path(tb.filename).name}:{tb.lineno}"})
            print(f"[error] {fid}: {exc} ({Path(tb.filename).name}:{tb.lineno})", flush=True)
        pd.DataFrame(rows).to_csv(out / "per_floor.csv", index=False)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"] if "status" in df else df
    summary = {"n_floors": int(len(ok)), "generator": gen_name, "status_counts": df["status"].str.split(":").str[0].value_counts().to_dict() if "status" in df else {}}
    for k in ["num_cycles", "avg_shortest_path", "diameter", "closeness_mean", "max_betweenness", "degree_entropy", "n_dead_ends", "sharp_angle_rate", "corridor_ratio", "n_entrances", "n_atria"]:
        b, aa = f"before_{k}", f"after_{k}"
        if b in ok and aa in ok and ok[aa].notna().any():
            d = (ok[aa] - ok[b]).dropna()
            summary[k] = {"before_mean": float(ok[b].mean()), "after_mean": float(ok[aa].mean()), "delta_mean": float(d.mean()) if len(d) else None,
                          "improved_rate": float(((d * BETTER.get(k, 0)) > 0).mean()) if len(d) and BETTER.get(k, 0) else None}
    if "after_crossings" in ok:
        summary["after_planar_rate"] = float((ok["after_crossings"] == 0).mean())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:2500])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
