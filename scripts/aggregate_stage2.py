#!/usr/bin/env python3
"""Aggregate stage-2 evaluation runs (<root>/<experiment>/[seed_k/]aggregate.json) into mean ± std tables."""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, pandas as pd

KEYS = ["overall_pass", "node_deviation_pct", "density_deviation_pct", "aspl_deviation_pct", "n_components", "target_edge_recall_pct", "target_edge_precision_pct",
        "attach_recall_pct", "attach_precision_pct", "degree_emd", "new_new_ratio_gen", "inference_time_s"]

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/experiments/stage2_eval_r4"); a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "aggregate.json")) + glob.glob(os.path.join(a.root, "*", "seed_*", "aggregate.json"))):
        rel = os.path.relpath(f, a.root).split(os.sep); exp = rel[0]; seed = rel[1] if len(rel) == 3 else "seed_0"
        d = json.load(open(f, encoding="utf-8")); rows.append({"experiment": exp, "seed": seed, **{k: d.get(k, np.nan) for k in KEYS}})
    if not rows:
        print("no runs"); return
    df = pd.DataFrame(rows).drop_duplicates(["experiment", "seed"]); df.to_csv(os.path.join(a.root, "per_seed.csv"), index=False)
    g = df.groupby("experiment"); m, s, n = g[KEYS].mean(), g[KEYS].std(ddof=0).fillna(0), g.size()
    summ = m.copy(); summ.insert(0, "seeds", n); summ.to_csv(os.path.join(a.root, "summary.csv"))
    (m.add_suffix("_mean").join(s.add_suffix("_std"))).to_csv(os.path.join(a.root, "summary_mean_std.csv"))
    order = ["ref_ground_truth", "stage2_rule_baseline", "stage2_search_baseline", "stage2_ar_gnn", "stage2_ar_gnn_greedy", "stage2_ar_gnn_bestof16",
             "stage2_ar_gnn_bfs_order", "stage2_ar_gnn_single_label", "stage2_ar_gnn_basic_feats", "stage2_ar_gnn_long", "stage2_ar_gnn_long_bestof16"]
    idx = [e for e in order if e in m.index] + [e for e in m.index if e not in order]
    lines = ["| experiment | seeds | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 2)]
    for e in idx:
        lines.append(f"| {e} | {n[e]} | " + " | ".join(f"{m.loc[e, k]:.3f} ± {s.loc[e, k]:.3f}" if n[e] > 1 else f"{m.loc[e, k]:.3f}" for k in KEYS) + " |")
    md = "\n".join(lines) + "\n"; open(os.path.join(a.root, "table.md"), "w", encoding="utf-8").write(md); print(md)

if __name__ == "__main__":
    main()
