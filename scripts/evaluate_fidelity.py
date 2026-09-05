#!/usr/bin/env python3
"""Prototype-fidelity protocol: model vs random / majority / oracle, plus layout predictability.

  python scripts/evaluate_fidelity.py --config configs/stage1/ridge.yaml [--configs cfg1 cfg2 ...]
"""
from __future__ import annotations
import json
from pathlib import Path
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.evaluation.prototype_fidelity import evaluate_prototype_fidelity, layout_predictability
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.utils import ProjectPaths, resolve_config, seed_everything, setup_logging

def main() -> None:
    p = base_parser("Prototype fidelity evaluation"); p.add_argument("--configs", nargs="*", default=[]); p.add_argument("--out", default="outputs/experiments/fidelity"); p.add_argument("--max-malls", type=int, default=None)
    a = p.parse_args(); setup_logging(a.log_level); paths = ProjectPaths(root=ROOT)
    cfgs = [a.config, *a.configs]; out = paths.resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    db = None; rows = []
    for i, c in enumerate(cfgs):
        cfg = resolve_config(paths.resolve(c), a.override); seed_everything(int(cfg.get("seed", 42)))
        if db is None: db = CaseDatabase.load(paths.resolve(cfg["data"]["processed_dir"]))
        pipe = Stage1Pipeline(cfg, db).fit()
        refs = [None, "random", "majority", "oracle"] if i == 0 else [None]
        for ref in refs:
            per_mall, agg = evaluate_prototype_fidelity(pipe, split=cfg["eval"].get("split", "test"), ks=(5, 10), max_malls=a.max_malls, reference=ref)
            name = cfg.get("experiment_name", Path(c).stem) if ref is None else f"ref_{ref}"
            per_mall.to_csv(out / f"{name}_per_mall.csv", index=False); rows.append({"name": name, **{k: v for k, v in agg.items() if not k.endswith("_std")}})
            print(f"{name:40s} " + " ".join(f"{k}={v:.3f}" for k, v in agg.items() if "@" in k and not k.endswith("_std")))
    lp = layout_predictability(db); (out / "layout_predictability.json").write_text(json.dumps(lp, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nlayout predictability (condition → type8):", json.dumps({k: v for k, v in lp.items() if k != "lgbm_feature_importance" and k != "classes"}, ensure_ascii=False))
    import pandas as pd; pd.DataFrame(rows).to_csv(out / "fidelity_summary.csv", index=False); print(f"written: {out}")

if __name__ == "__main__":
    main()
