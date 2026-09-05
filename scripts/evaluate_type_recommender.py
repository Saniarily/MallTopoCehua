#!/usr/bin/env python3
"""Evaluate the type-conditional quality model E[score | conditions, type] on held-out malls."""
from __future__ import annotations
import json
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.stage1.type_recommender import TreeTypeRecommender, evaluate_type_recommender
from mall_space_planner.utils import ProjectPaths, resolve_config, seed_everything, setup_logging

def main() -> None:
    p = base_parser("Type recommender evaluation"); p.add_argument("--out", default="outputs/experiments/type_recommender"); p.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    a = p.parse_args(); setup_logging(a.log_level); cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    db = CaseDatabase.load(paths.resolve(cfg["data"]["processed_dir"])); out = paths.resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    params = {k: v for k, v in cfg.get("stage1", {}).get("type_recommender", {}).items() if k != "enabled"}
    results = []
    for seed in a.seeds:
        seed_everything(seed); rec = TreeTypeRecommender(**{**params, "seed": seed}).fit(db, db.split("train")); r = evaluate_type_recommender(rec, db); r["seed"] = seed; results.append(r)
        print(f"[seed {seed}] rmse type/cond={r['rmse_with_type']:.4f}/{r['rmse_conditions_only']:.4f} spearman type/cond={r['spearman_with_type']:.3f}/{r['spearman_conditions_only']:.3f} tau_type_order={r['mean_kendall_tau_type_order']:.3f} best_type_agree={r['best_type_agreement_rate']:.2f} policy_uplift={r['policy_uplift']:.4f}")
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("per-cluster (last seed):"); [print("  ", c) for c in results[-1]["per_cluster"]]; print(f"written: {out}")

if __name__ == "__main__":
    main()
