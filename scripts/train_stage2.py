#!/usr/bin/env python3
"""Train the AR-GNN topology expander on a skeleton→topology corpus and save a checkpoint.

  python scripts/train_stage2.py --config configs/stage2/ar_gnn.yaml --corpus data/processed/legacy/stage2_corpus_v2.jsonl   # (v1: /path/sharegpt_data.json)
"""
from __future__ import annotations
import json
from _common import ROOT, base_parser
from mall_space_planner.data.corpus_builder import load_any_corpus
from mall_space_planner.registry import build
from mall_space_planner.utils import ProjectPaths, resolve_config, seed_everything, setup_logging

def main() -> None:
    p = base_parser("Train stage-2 generator"); p.add_argument("--corpus", default=None); p.add_argument("--limit", type=int, default=None); p.add_argument("--force", action="store_true")
    a = p.parse_args(); setup_logging(a.log_level); cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT); seed_everything(int(cfg.get("seed", 42)))
    corpus = a.corpus or cfg.get("corpus", "data/samples/synthetic/sharegpt_sample.json")
    ck = paths.resolve(cfg.get("checkpoint_dir", "outputs/checkpoints/stage2")) / cfg.get("experiment_name", "ar_gnn")
    if (ck / "ar_gnn.pt").exists() and not a.force:
        print(f"cached checkpoint: {ck} (use --force to retrain)"); return
    cp = paths.resolve(corpus)
    if cp.suffix.lower() == ".jsonl":  # corpus v2: mall-grouped split stored in the file
        train = load_any_corpus(cp, split="train", limit=a.limit or cfg.get("train_limit"))
    else:  # corpus v1 (ShareGPT): hold out the last `eval_holdout` samples (file order) so train/eval never overlap
        samples = load_any_corpus(cp, limit=a.limit or cfg.get("train_limit")); hold = int(cfg.get("eval_holdout", 0)); train = samples[:-hold] if hold else samples
    gen = build("generator", cfg["stage2"]["generator"]); gen.fit(train)
    out = paths.resolve(cfg.get("checkpoint_dir", "outputs/checkpoints/stage2")) / cfg.get("experiment_name", "ar_gnn"); gen.save(out)
    hist = gen.history_; print(json.dumps({k: (round(v[-1], 4) if v else None) for k, v in hist.items()}, indent=1)); print(f"checkpoint: {out}")

if __name__ == "__main__":
    main()
