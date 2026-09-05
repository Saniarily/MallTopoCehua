#!/usr/bin/env python3
"""Fit a Stage-1 pipeline from config, evaluate on val, save checkpoint."""
from __future__ import annotations
import json
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.evaluation.stage1_eval import evaluate_stage1
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.utils import ProjectPaths, resolve_config, seed_everything, setup_logging

def main() -> None:
    a = base_parser("Train stage 1").parse_args(); setup_logging(a.log_level)
    cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT); seed_everything(int(cfg.get("seed", 42)))
    db = CaseDatabase.load(paths.resolve(cfg["data"]["processed_dir"]))
    pipe = Stage1Pipeline(cfg, db).fit()
    name = cfg.get("experiment_name", "stage1"); out = paths.resolve(cfg.get("output_dir", "outputs/experiments/stage1")) / name
    pipe.save(out / "checkpoint")
    _, agg = evaluate_stage1(pipe, split="val", ks=tuple(cfg["eval"].get("ks", (5, 10))), min_candidates=cfg["eval"].get("min_candidates", 5))
    (out / "val_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in agg.items() if not k.endswith("_std")}, indent=2)); print(f"checkpoint: {out / 'checkpoint'}")

if __name__ == "__main__":
    main()
