#!/usr/bin/env python3
"""Stage-3 evaluation over the real corpus: fit each floor's complete M network (topology only) into an outline
and compare with where the real M junctions are.

Two protocols per floor (test split of corpus v2 by default):
  self      – fit into the floor's OWN outline (mask PNG if configured, else *_total.csv polygons) → chamfer /
              procrustes vs real CenterPoints, plus geometric sanity (crossings, ortho, corridor ratio vs the real
              area_corridor/area_total from dataset_0.csv);
  transfer  – fit into the outline of the closest-area OTHER mall (what the web UI does) → geometric sanity only.

  python scripts/evaluate_stage3.py --config configs/data/legacy.yaml --limit 100
  python scripts/evaluate_stage3.py --config configs/data/legacy.yaml --split test --out outputs/experiments/stage3_eval

Outputs: per_floor.csv + summary.json (means, medians, random baselines) in --out.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from _common import ROOT, base_parser

from mall_space_planner.data.corpus_builder import load_corpus_jsonl, load_target_csv
from mall_space_planner.data.legacy_adapter import split_floor_id
from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, render_corridors
from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths
from mall_space_planner.stage3.evaluate import evaluate_fit
from mall_space_planner.stage3.outline import m_positions_from_total_csv
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

KEEP = ["crossings", "inside_ratio", "ortho_deviation_deg", "sharp_angle_rate", "spacing_violation_rate", "served_area_ratio", "corridor_ratio", "n_entrances", "n_atria",
        "outer_facade_dist_m", "gt_outer_facade_dist_m", "chamfer_m", "chamfer_rand_m", "procrustes_rmse_m", "procrustes_rmse_rand_m", "procrustes_rmse_norm"]


def main() -> None:
    p = base_parser("Evaluate Stage 3 (corridor adaptation) on real floors")
    p.add_argument("--corpus", default=None); p.add_argument("--split", default="test"); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="outputs/experiments/stage3_eval"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-transfer", action="store_true"); p.add_argument("--restarts", type=int, default=4); p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-nodes", type=int, default=120, help="skip floors with more M nodes (fit cost grows ~n^2); they are counted as 'skipped_large'")
    p.add_argument("--timeout", type=float, default=300.0, help="per-floor wall-clock limit in seconds (self + transfer); over-limit floors are recorded as 'timeout'")
    p.add_argument("--resume", action="store_true", help="skip floors already present in <out>/per_floor.csv")
    a = p.parse_args(); setup_logging(a.log_level); cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    try:
        import PIL, skimage  # noqa: F401  (fail fast instead of 200 identical error rows)
    except ImportError as exc:
        raise SystemExit(f"Stage 3 needs Pillow + scikit-image: pip install pillow scikit-image  ({exc})") from exc
    s3 = Stage3Paths.from_config(cfg); ds = Stage3Dataset(s3)
    if s3.graph_dir is None:
        raise SystemExit("dataset.params.graph_dir missing in config")
    corpus = paths.resolve(a.corpus or cfg.get("stage2_corpus", "data/processed/legacy/stage2_corpus_v2.jsonl"))
    samples = load_corpus_jsonl(corpus, split=a.split or None, limit=a.limit)
    shop_depth = float((cfg.get("stage3") or {}).get("shop_depth_m", 14.0)); cr = float((cfg.get("stage3") or {}).get("corridor_ratio", 0.18))
    fitter = CorridorFitter(FitParams(shop_depth=shop_depth, n_restarts=a.restarts, iters=a.iters)); rp = RenderParams(corridor_ratio=cr)
    out = paths.resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []; t0 = time.time()
    per_floor_csv = out / "per_floor.csv"
    done: set[str] = set()
    if a.resume and per_floor_csv.exists():
        prev = pd.read_csv(per_floor_csv)
        # keep only floors that finished (ok / deliberately skipped); error and timeout rows are dropped and retried
        keep_status = {"ok", "skipped_large", "no_total_csv"}
        finished = set(prev.loc[prev["status"].isin(keep_status), "floor_id"].astype(str))
        prev = prev[prev["floor_id"].astype(str).isin(finished)]
        rows = prev.to_dict("records"); done = finished
        n_retry = int((~pd.read_csv(per_floor_csv)["floor_id"].astype(str).isin(finished)).sum())
        print(f"resume: {len(done)} floors already evaluated, {n_retry} error/timeout rows will be retried", flush=True)

    def _flush() -> None:
        pd.DataFrame(rows).to_csv(per_floor_csv, index=False)

    for i, smp in enumerate(samples):
        fid = smp.sample_id; mall, _ = split_floor_id(fid)
        if fid in done:
            continue
        tp = s3.total_csv(fid)
        if tp is None:
            rows.append({"floor_id": fid, "protocol": "self", "status": "no_total_csv"}); continue
        t_floor = time.time()
        try:
            outline = ds.outline(fid, 0, area_m2=None)
            topo = load_target_csv(tp.with_name(fid + "_M.csv")) if (tp.with_name(fid + "_M.csv")).exists() else smp.target
            if topo.num_nodes > a.max_nodes:
                rows.append({"floor_id": fid, "mall_id": mall, "protocol": "self", "status": "skipped_large", "n_nodes": topo.num_nodes}); continue
            print(f"  [{i + 1}/{len(samples)}] {fid}: {topo.num_nodes} nodes, outline {outline.area:.0f} m2", flush=True)
            gt = m_positions_from_total_csv(tp, outline)
            res = fitter.fit(topo, outline, seed=a.seed); plan = render_corridors(topo, res.positions, outline, res.roles, rp)
            ev = evaluate_fit(topo, res, plan, outline, gt_positions={k: gt[k] for k in topo.nodes if k in gt})
            row = {"floor_id": fid, "mall_id": mall, "protocol": "self", "status": "ok", "n_nodes": topo.num_nodes, "outline_area_m2": outline.area,
                   "scale_source": outline.scale_source, "real_corridor_ratio": outline.extra.get("real_corridor_ratio"), **{k: ev.get(k) for k in KEEP}}
            row["fit_seconds"] = round(time.time() - t_floor, 1)
            rows.append(row)
            if time.time() - t_floor > a.timeout:
                rows.append({"floor_id": fid, "mall_id": mall, "protocol": "transfer", "status": "timeout", "n_nodes": topo.num_nodes})
            elif not a.no_transfer:
                cands = ds.similar_outlines(outline.area, k=1, exclude_mall=mall)
                if not cands:
                    rows.append({"floor_id": fid, "mall_id": mall, "protocol": "transfer", "status": "no_candidate_outline", "n_nodes": topo.num_nodes})
                if cands:
                    o2 = ds.outline(cands[0], 0); res2 = fitter.fit(topo, o2, seed=a.seed); plan2 = render_corridors(topo, res2.positions, o2, res2.roles, rp)
                    ev2 = evaluate_fit(topo, res2, plan2, o2)
                    rows.append({"floor_id": fid, "mall_id": mall, "protocol": "transfer", "status": "ok", "target_outline": cands[0], "n_nodes": topo.num_nodes,
                                 "outline_area_m2": o2.area, "real_corridor_ratio": o2.extra.get("real_corridor_ratio"), **{k: ev2.get(k) for k in KEEP if k in ev2}})
        except Exception as exc:  # noqa: BLE001
            import traceback

            tb = traceback.extract_tb(exc.__traceback__)[-1]
            rows.append({"floor_id": fid, "protocol": "self", "status": f"error: {type(exc).__name__}: {exc}"[:200], "error_at": f"{Path(tb.filename).name}:{tb.lineno}"})
            print(f"  !! {fid}: {type(exc).__name__}: {exc} ({Path(tb.filename).name}:{tb.lineno})", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)} floors  ({time.time() - t0:.0f}s)", flush=True); _flush()
    _flush(); df = pd.DataFrame(rows)
    summary: dict = {"n_samples": len(samples), "split": a.split, "corpus": str(corpus), "status_counts": df["status"].value_counts().to_dict(), "corridor_ratio_real_dataset": ds.corridor_ratio_stats()}
    for proto in ("self", "transfer"):
        sub = df[(df["protocol"] == proto) & (df["status"] == "ok")]
        if sub.empty:
            continue
        num = sub.select_dtypes("number")
        summary[proto] = {"n": int(len(sub)), "mean": {k: round(float(v), 3) for k, v in num.mean().items()}, "median": {k: round(float(v), 3) for k, v in num.median().items()},
                          "planar_rate": float((sub["crossings"] == 0).mean()), "all_inside_rate": float((sub["inside_ratio"] >= 0.999).mean())}
        if "chamfer_m" in sub and sub["chamfer_m"].notna().any():
            summary[proto]["chamfer_better_than_random_rate"] = float((sub["chamfer_m"] < sub["chamfer_rand_m"]).mean())
            summary[proto]["chamfer_ratio_to_random"] = float((sub["chamfer_m"] / sub["chamfer_rand_m"]).median())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "status_counts"} | {"status_counts": summary["status_counts"]}, indent=1, ensure_ascii=False, default=str)); print(f"written: {out}")


if __name__ == "__main__":
    main()
