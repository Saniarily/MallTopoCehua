#!/usr/bin/env python3
"""Build the processed CaseDatabase from a data config and write an audit report."""
from __future__ import annotations
import json
from _common import ROOT, base_parser
from mall_space_planner.data.audit import audit_case_database, render_markdown
from mall_space_planner.registry import build
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

def main() -> None:
    a = base_parser("Prepare data").parse_args(); setup_logging(a.log_level)
    cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    ds = cfg["dataset"]; params = dict(ds.get("params", {}))
    if "out_dir" in params: params["out_dir"] = str(paths.resolve(params["out_dir"]))
    db = build("dataset_adapter", {"name": ds["adapter"], "params": params}).build()
    out = paths.resolve(cfg.get("processed_dir", "data/processed/default")); db.save(out)
    rep = audit_case_database(db)
    (out / "audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "audit.md").write_text(render_markdown(f"Data audit: {rep.get('source')}", rep), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("source", "synthetic", "n_rows", "n_malls", "n_graphs", "split_counts", "leakage") if k in rep}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
