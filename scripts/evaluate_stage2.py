#!/usr/bin/env python3
"""Evaluate a Stage-2 generator config on the skeleton→topology corpus (v2 JSONL from CSVs, v1 ShareGPT, or synthetic).

For each sample: skeleton → generate with N_target from the prompt → spec metrics vs. skeleton
and (optionally) vs. the ground-truth expansion. Writes per-sample CSV + aggregate JSON.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from _common import ROOT, base_parser
from mall_space_planner.data.corpus_builder import load_any_corpus
from mall_space_planner.evaluation.stage2_eval import TopologySpecEvaluator
from mall_space_planner.registry import build
from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

def main() -> None:
    p = base_parser("Evaluate stage 2 on skeleton→topology corpus"); p.add_argument("--corpus", default="data/samples/synthetic/sharegpt_sample.json"); p.add_argument("--limit", type=int, default=200); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ground-truth", action="store_true", help="score the corpus' own complete topologies instead of a generator (reference row)")
    p.add_argument("--force", action="store_true", help="recompute even if aggregate.json exists")
    a = p.parse_args(); setup_logging(a.log_level); cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    corpus = a.corpus if a.corpus != p.get_default("corpus") else cfg.get("corpus", a.corpus)
    _name = "ref_ground_truth" if a.ground_truth else cfg.get("experiment_name", "stage2"); _out = paths.resolve(cfg.get("eval_output_dir", "outputs/experiments/stage2_eval")) / _name / (f"seed_{a.seed}" if a.seed else "")
    if (_out / "aggregate.json").exists() and not a.force:
        print(f"cached: {_out}"); return
    cp = paths.resolve(corpus)
    if cp.suffix.lower() == ".jsonl":  # corpus v2: evaluate on the mall-grouped test split
        samples = load_any_corpus(cp, split=cfg.get("eval_split", "test"))[: a.limit]
    else:  # corpus v1: evaluation set = held-out tail (never used for training)
        all_samples = load_any_corpus(cp); hold = int(cfg.get("eval_holdout", 0)); samples = (all_samples[-hold:] if hold else all_samples)[: a.limit]
    gen = None if a.ground_truth else build("generator", cfg["stage2"]["generator"]); ev: TopologySpecEvaluator = build("evaluator", cfg["stage2"].get("evaluator", {"name": "topology_spec"}))
    rows = []
    for s in samples:
        n_t = s.target_num_nodes or s.target.num_nodes
        req = GenerationRequest(prototype=TopologyPrototype(prototype_id=s.sample_id, graph=s.skeleton, layout_type=s.layout_type), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=n_t, layout_type=s.layout_type), seed=a.seed)
        t0 = time.perf_counter(); g = s.target if gen is None else gen.generate(req, a.seed); dt = time.perf_counter() - t0
        r = ev.evaluate(s.skeleton, g, n_t, inference_time_s=dt, target=s.target)
        rows.append({"sample_id": s.sample_id, "layout": s.layout_type.value if s.layout_type else None, "n_skeleton": s.skeleton.num_nodes, "n_target": n_t, "overall_pass": r.overall_pass, **{k: v for k, v in r.metrics.items()}, **{f"pass_{k}": v for k, v in r.passed.items()}})
    df = pd.DataFrame(rows); name = "ref_ground_truth" if a.ground_truth else cfg.get("experiment_name", "stage2"); out = paths.resolve(cfg.get("eval_output_dir", "outputs/experiments/stage2_eval")) / name / (f"seed_{a.seed}" if a.seed else ""); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_sample.csv", index=False)
    num = df.select_dtypes(include=[np.number, bool]); agg = {"n_samples": int(len(df)), "generator": "ground_truth" if a.ground_truth else cfg["stage2"]["generator"]["name"], **{c: float(num[c].mean()) for c in num.columns}}
    agg["by_layout"] = df.groupby("layout")[["overall_pass", "node_deviation_pct", "density_deviation_pct", "aspl_deviation_pct"]].mean().round(3).to_dict("index")
    (out / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items() if k != "by_layout"}, ensure_ascii=False, indent=1)); print(f"written: {out}")

if __name__ == "__main__":
    main()
