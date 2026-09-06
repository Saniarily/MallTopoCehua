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
    # markdown: seed-averaged summary + per-cluster type tables (last seed; CI is bootstrap, not seed)
    import numpy as np
    keys = ["rmse_with_type", "rmse_conditions_only", "spearman_with_type", "spearman_conditions_only", "mean_kendall_tau_type_order", "best_type_agreement_rate", "top1_separable_rate", "policy_uplift"]
    md = ["# Type-conditional quality model E[score | conditions, type]", "", f"seeds: {a.seeds}; test floors: {results[0]['n_test_floors']}, test malls: {results[0]['n_test_malls']}", "", "| metric | mean ± std over seeds |", "|---|---|"]
    md += [f"| {k} | {np.mean([r[k] for r in results]):.4f} ± {np.std([r[k] for r in results]):.4f} |" for k in keys if all(k in r for r in results)]
    for c, rows in results[-1].get("type_tables", {}).items():
        md += ["", f"## cluster {c}", "", "| rank | layout_type | E[score] | 10-90% CI | empirical mean | n |", "|---|---|---|---|---|---|"]
        md += [f"| {r['rank']} | {r['layout_type']} | {r['expected_score']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {'-' if r['empirical_mean'] is None else format(r['empirical_mean'], '.3f')} | {r['empirical_n']} |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
    print(f"written: {out}")

if __name__ == "__main__":
    main()
