#!/usr/bin/env python3
"""Evaluate a saved Stage-1 checkpoint on the test split; export CSV/JSON."""
from __future__ import annotations
import json
from pathlib import Path
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.evaluation.stage1_eval import evaluate_stage1
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

def main() -> None:
    p = base_parser("Evaluate stage 1"); p.add_argument("--checkpoint", required=True); a = p.parse_args(); setup_logging(a.log_level)
    cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    db = CaseDatabase.load(paths.resolve(cfg["data"]["processed_dir"]))
    pipe = Stage1Pipeline.load(paths.resolve(a.checkpoint), db, cfg)
    per_q, agg = evaluate_stage1(pipe, split=cfg["eval"].get("split", "test"), ks=tuple(cfg["eval"].get("ks", (5, 10, 20))), min_candidates=cfg["eval"].get("min_candidates", 5))
    out = Path(paths.resolve(a.checkpoint)).parent
    per_q.to_csv(out / "test_per_query.csv", index=False); (out / "test_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in agg.items() if not k.endswith("_std")}, indent=2))

if __name__ == "__main__":
    main()
