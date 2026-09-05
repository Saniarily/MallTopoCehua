#!/usr/bin/env python3
"""Multi-seed ablation / model-comparison driver.

Config format (configs/ablations/*.yaml):
  base_config: configs/stage1/lgbm_lambdarank.yaml
  seeds: [42, 43, 44]
  factors:                       # each factor = one variant; overrides applied on top of base
    - {name: full, overrides: {}}
    - {name: no_condition, overrides: {features.use_condition: false}}
  compare_configs:               # optional: whole configs to compare (model comparison)
    - configs/stage1/random_forest.yaml
"""
from __future__ import annotations
import datetime as dt, json
from pathlib import Path
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.experiments.aggregate import aggregate_runs, plot_metric_bars, to_markdown_table
from mall_space_planner.experiments.runner import run_stage1_experiment
from mall_space_planner.utils import ProjectPaths, load_yaml, resolve_config, setup_logging

def main() -> None:
    p = base_parser("Run ablation"); p.add_argument("--out-dir", default=None); a = p.parse_args(); setup_logging(a.log_level)
    ab = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S"); out = paths.resolve(a.out_dir or f"outputs/experiments/ablation_{Path(a.config).stem}_{stamp}"); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in ab.get("seeds", [42])]
    jobs: list[tuple[str, dict]] = []
    if ab.get("base_config"):
        base = paths.resolve(ab["base_config"])
        for f in ab.get("factors", [{"name": "full", "overrides": {}}]):
            ov = [f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in (f.get("overrides") or {}).items()]
            cfg = resolve_config(base, ov); cfg["variant"] = f["name"]; jobs.append((f["name"], cfg))
    for c in ab.get("compare_configs", []):
        cfg = resolve_config(paths.resolve(c)); cfg["variant"] = "compare"; jobs.append((Path(c).stem, cfg))
    db_cache: dict[str, CaseDatabase] = {}
    for name, cfg in jobs:
        pdir = str(paths.resolve(cfg["data"]["processed_dir"])); db = db_cache.setdefault(pdir, CaseDatabase.load(pdir))
        for seed in seeds:
            rdir = out / name / f"seed_{seed}"
            rec = run_stage1_experiment(cfg, db, rdir, seed=seed, save_checkpoint=False); rec["variant"] = cfg.get("variant", "")
            (rdir / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            t = rec["metrics"].get("test", {}); print(f"[{name}][seed {seed}] test ndcg@5={t.get('ndcg@5', float('nan')):.4f} ndcg@10={t.get('ndcg@10', float('nan')):.4f} spearman={t.get('spearman', float('nan')):.4f}")
    for split in ("val", "test"):
        agg = aggregate_runs(out, split=split)
        if agg.empty: continue
        agg.to_csv(out / f"summary_{split}.csv", index=False); (out / f"table_{split}.md").write_text(to_markdown_table(agg), encoding="utf-8")
        plot_metric_bars(agg, "ndcg@10", out / f"ndcg10_{split}.png")
    print((out / "table_test.md").read_text(encoding="utf-8") if (out / "table_test.md").exists() else "no test table"); print(f"results: {out}")

if __name__ == "__main__":
    main()
