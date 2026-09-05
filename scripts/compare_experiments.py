#!/usr/bin/env python3
"""Aggregate all run.json under a root into a paper table + bar charts."""
from __future__ import annotations
import argparse
from _common import ROOT
from mall_space_planner.experiments.aggregate import aggregate_runs, plot_metric_bars, to_markdown_table
from mall_space_planner.utils import ProjectPaths

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", default="outputs/experiments"); p.add_argument("--split", default="test"); p.add_argument("--out", default="outputs/reports/comparison")
    a = p.parse_args(); paths = ProjectPaths(root=ROOT); out = paths.resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    agg = aggregate_runs(paths.resolve(a.root), split=a.split)
    agg.to_csv(out / f"comparison_{a.split}.csv", index=False); md = to_markdown_table(agg); (out / f"comparison_{a.split}.md").write_text(md, encoding="utf-8")
    for m in ("ndcg@5", "ndcg@10", "spearman"): plot_metric_bars(agg, m, out / f"{m.replace('@', '_at_')}_{a.split}.png")
    print(md); print(f"written to {out}")

if __name__ == "__main__":
    main()
