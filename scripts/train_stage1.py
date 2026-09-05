#!/usr/bin/env python3
"""Fit a Stage-1 pipeline from config, evaluate on val+test, save checkpoint and run.json."""
from __future__ import annotations
import json
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.experiments.runner import run_stage1_experiment
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

def main() -> None:
    p = base_parser("Train stage 1"); p.add_argument("--seed", type=int, default=None); a = p.parse_args(); setup_logging(a.log_level)
    cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    db = CaseDatabase.load(paths.resolve(cfg["data"]["processed_dir"]))
    name = cfg.get("experiment_name", "stage1"); seed = a.seed if a.seed is not None else int(cfg.get("seed", 42))
    out = paths.resolve(cfg.get("output_dir", "outputs/experiments/stage1")) / name / f"seed_{seed}"
    rec = run_stage1_experiment(cfg, db, out, seed=seed)
    for split, m in rec["metrics"].items():
        print(split, json.dumps({k: round(v, 4) for k, v in m.items() if isinstance(v, float) and not k.endswith("_std")}))
    print(f"checkpoint: {out / 'checkpoint'}")

if __name__ == "__main__":
    main()
