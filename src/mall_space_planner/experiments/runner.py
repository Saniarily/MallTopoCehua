"""Single Stage-1 experiment run = fit + val eval + test eval + checkpoint + JSON record.

Every run writes ``run.json`` with config, seed, metrics, timing, feature importance and
environment snapshot so that :mod:`.aggregate` can pool runs across seeds/variants.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.evaluation.stage1_eval import evaluate_stage1
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.utils.logging import get_logger
from mall_space_planner.utils.repro import collect_environment_info, seed_everything

logger = get_logger(__name__)


def run_stage1_experiment(cfg: dict[str, Any], db: CaseDatabase, out_dir: Path, seed: int | None = None, splits: tuple[str, ...] = ("val", "test"), save_checkpoint: bool = True) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42) if seed is None else seed)
    cfg = dict(cfg, seed=seed)
    if "stage1" in cfg and isinstance(cfg["stage1"].get("ranker"), dict):
        cfg["stage1"]["ranker"].setdefault("params", {})["seed"] = seed if cfg["stage1"]["ranker"]["name"] != "weighted_rule" else cfg["stage1"]["ranker"]["params"].get("seed", None)
        if cfg["stage1"]["ranker"]["params"].get("seed") is None:
            cfg["stage1"]["ranker"]["params"].pop("seed", None)
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pipe = Stage1Pipeline(cfg, db).fit()
    fit_s = time.perf_counter() - t0
    ev_cfg = cfg.get("eval", {})
    record: dict[str, Any] = {"experiment_name": cfg.get("experiment_name", "stage1"), "seed": seed, "ranker": pipe.fit_info.get("ranker"), "fit_seconds": fit_s, "config": cfg, "metrics": {}}
    for split in splits:
        try:
            per_q, agg = evaluate_stage1(pipe, split=split, ks=tuple(ev_cfg.get("ks", (5, 10, 20))), binary_top_frac=float(ev_cfg.get("binary_top_frac", 0.2)), min_candidates=int(ev_cfg.get("min_candidates", 5)))
            per_q.to_csv(out_dir / f"{split}_per_query.csv", index=False)
            record["metrics"][split] = agg
        except ValueError as exc:
            logger.warning("Skipping %s eval: %s", split, exc)
    record["feature_importance"] = pipe.ranker.feature_importance()
    record["history"] = pipe.ranker.training_history()
    record["environment"] = collect_environment_info()
    if save_checkpoint:
        pipe.save(out_dir / "checkpoint")
    (out_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return record
