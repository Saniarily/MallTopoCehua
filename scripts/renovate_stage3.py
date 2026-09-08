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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mall_space_planner.data.corpus_builder import load_target_csv  # noqa: E402
from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams  # noqa: E402
from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths  # noqa: E402
from mall_space_planner.stage3.renovate import BETTER, build_generator, make_score_fn, renovate_floor, select_floors  # noqa: E402
from mall_space_planner.utils.config import resolve_config  # noqa: E402

def draw(r: dict, out_png: Path, gen_name: str, title: str) -> None:  # noqa: ANN001
    """Four-panel figure (shared with the Viewer Hub: mall_space_planner.hub.viz.draw_renovation)."""
    import matplotlib.pyplot as plt

    from mall_space_planner.hub.viz import draw_renovation

    fig = draw_renovation(r, gen_name, title)
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
    ap.add_argument("--min-nodes", type=int, default=12, help="floor filter: at least this many key points")
    ap.add_argument("--min-area", type=float, default=6000.0, help="floor filter: at least this floor area (m2)")
    ap.add_argument("--no-filter", action="store_true", help="take --floors verbatim (no size / entrance / floor-index filter)")
    ap.add_argument("--keep-skeleton-positions", action="store_true", help="anchor all skeleton nodes at their real positions (old behaviour); default anchors only the existing entrances")
    ap.add_argument("--no-score", action="store_true", help="do not use the Stage-1 predicted score in candidate selection")
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
    cases = None
    cases_p = Path(cfg.get("data", {}).get("processed_dir", "data/processed/legacy")) / "cases.csv"
    if cases_p.exists():
        cases = pd.read_csv(cases_p)
    elif a.low_score is not None:
        print(f"[warn] {cases_p} not found; --low-score ignored")
    if not a.no_filter:
        n0 = len(floors)
        floors = select_floors(ds, s3.graph_dir, floors, cases=cases, low_score=a.low_score, min_nodes=a.min_nodes, min_area_m2=a.min_area, max_nodes=a.max_nodes, require_entrance=True, prefer_floors=(1, 2))
        print(f"floor filter: {n0} -> {len(floors)} (score<= {a.low_score}, nodes>= {a.min_nodes}, area>= {a.min_area:.0f} m2, floors 1-2 with an entrance)")
    floors = floors[: a.limit] if a.limit else floors
    if not floors:
        raise SystemExit("no floors selected")
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen, gen_name = build_generator(Path(a.results))
    svc, db = None, None
    if not a.no_score:
        try:
            from mall_space_planner.api.service import PlanningService
            from mall_space_planner.data.case_db import CaseDatabase

            s1 = resolve_config("configs/stage1/extra_trees.yaml", []); s2 = resolve_config("configs/stage2/search_baseline.yaml", [])
            proc = Path(s1["data"]["processed_dir"])
            if (proc / "manifest.json").exists():
                db = CaseDatabase.load(str(proc)); s1["stage1"]["counterfactuals"] = {"enabled": False}
                svc = PlanningService(db, s1, s2)
                print("stage-1 scorer ready (predicted score used in candidate selection)")
            else:
                print(f"[warn] {proc} has no manifest.json; predicted score disabled")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] stage-1 scorer unavailable ({exc}); predicted score disabled")
    shop_depth = float((cfg.get("stage3") or {}).get("shop_depth_m", 14.0)); cr = float((cfg.get("stage3") or {}).get("corridor_ratio", 0.18))
    fitter = CorridorFitter(FitParams(shop_depth=shop_depth, n_restarts=a.restarts, iters=a.iters)); rp = RenderParams(corridor_ratio=cr)
    rows = []
    for i, fid in enumerate(floors):
        t0 = time.time()
        try:
            n_probe = load_target_csv(s3.graph_dir / f"{fid}_M.csv").num_nodes if (s3.graph_dir / f"{fid}_M.csv").exists() else None
            if n_probe is not None and n_probe > a.max_nodes:
                rows.append({"floor_id": fid, "status": "skipped_large", "n_nodes": n_probe}); continue
            score_fn = None
            if svc is not None:
                from mall_space_planner.schemas import PlanningCondition

                row_c = db.cases[db.cases[db.id_col] == fid]
                if not row_c.empty:
                    r0 = row_c.iloc[0]
                    cond = PlanningCondition(city_cluster=int(r0["city_cluster"]) if pd.notna(r0.get("city_cluster")) else None, **{c: (float(r0[c]) if pd.notna(r0.get(c)) else None) for c in db.query_cols})
                    score_fn = make_score_fn(svc, cond)
            r = renovate_floor(fid, s3.graph_dir, ds, gen, fitter, rp, seed=a.seed, n_candidates=a.candidates, score_fn=score_fn, keep_skeleton_positions=a.keep_skeleton_positions)
            draw(r, out / f"{fid}.png", gen_name, f"旧商场改造：{fid}（同一轮廓、保留出入口位置，以原型为骨架重新生长并重排布局）")
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
    for k in ["pred_score", "num_cycles", "avg_shortest_path", "diameter", "closeness_mean", "max_betweenness", "degree_entropy", "n_dead_ends", "sharp_angle_rate", "corridor_ratio", "n_entrances", "n_atria"]:
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
