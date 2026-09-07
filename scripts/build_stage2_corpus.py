#!/usr/bin/env python3
"""Build the real Stage-2 corpus (v2) from per-floor CSV graph files.

  python scripts/build_stage2_corpus.py --config configs/data/legacy.yaml \
      [--graph-dir /path/to/total_graph_data] [--out data/processed/legacy/stage2_corpus_v2.jsonl] [--limit N]

Reads ``{floor_id}_M_simplified.csv`` + ``..._node_attributes.csv`` (skeleton) and ``{floor_id}_M.csv``
(complete corridor topology) for every floor found in ``graph_dir``; joins conditions from the main
table; assigns a mall-grouped, cluster-stratified train/val/test split (same protocol as Stage 1);
writes JSONL + a stats JSON next to it. See ``docs/data_audit.md`` §Stage-2 v2.
"""
from __future__ import annotations

import json
from pathlib import Path

from _common import ROOT, base_parser
from mall_space_planner.data.corpus_builder import build_corpus
from mall_space_planner.data.legacy_adapter import LegacyDataSpec, load_main_table
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging


def main() -> None:
    p = base_parser("Build Stage-2 corpus v2 from CSV graph files")
    p.add_argument("--graph-dir", default=None, help="folder with *_M.csv / *_M_simplified*.csv (default: dataset.params.graph_dir)")
    p.add_argument("--out", default=None, help="output JSONL (default: <processed_dir>/stage2_corpus_v2.jsonl)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-main-table", action="store_true", help="do not join conditions from the main table")
    a = p.parse_args()
    setup_logging(a.log_level)
    cfg = resolve_config(a.config, a.override)
    paths = ProjectPaths(root=ROOT)
    dp = (cfg.get("dataset", {}) or {}).get("params", {}) or {}
    feats = dp.get("features", {}) or {}
    spec = LegacyDataSpec(
        main_table_csv=Path(dp.get("main_table_csv", "/nonexistent")).expanduser(),
        graph_dir=Path(dp.get("graph_dir", ".")).expanduser(),
        **{k: v for k, v in feats.items() if k in LegacyDataSpec.__dataclass_fields__},
    )
    graph_dir = Path(a.graph_dir) if a.graph_dir else Path(spec.graph_dir)
    out = Path(a.out) if a.out else paths.resolve(cfg.get("processed_dir", "data/processed/legacy")) / "stage2_corpus_v2.jsonl"
    main_table = None
    if not a.no_main_table:
        try:
            main_table = load_main_table(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] main table not available ({exc}); conditions will be empty")
    split = dp.get("split", {}) or {}
    path, stats = build_corpus(
        graph_dir, out, main_table,
        id_col=spec.id_col, mall_id_col=spec.mall_id_col, cluster_col=spec.city_cluster_col,
        val_ratio=float(split.get("val_ratio", 0.1)), test_ratio=float(split.get("test_ratio", 0.1)), seed=int(split.get("seed", 42)),
        limit=a.limit,
    )
    (path.with_suffix(".stats.json")).write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    print(json.dumps(stats.as_dict(), indent=1))
    print(f"corpus: {path}")


if __name__ == "__main__":
    main()
