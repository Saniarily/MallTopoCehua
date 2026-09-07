#!/usr/bin/env python3
"""Calibrate the Stage-2 outline thresholds on the *ground-truth* row of a corpus.

The five outline metrics compare a generated topology with its skeleton (density / ASPL deviation
etc.). Their pass thresholds were first set on corpus v1 (LLM-generated targets). On corpus v2 the
real built topologies are much denser than their skeletons (sample B000A0E928_1: 28 → 83 edges), so
the ground truth itself would fail a 40 % density-deviation threshold. This script sets every
``*_max`` threshold to the ``--quantile`` (default 0.95) of the ground-truth distribution, so that
"pass" means *within the range of real buildings*; it writes a YAML override you pass to the
evaluations::

  python scripts/evaluate_stage2.py --config configs/stage2/rule_baseline.yaml --corpus C.jsonl --ground-truth --override eval_output_dir=OUT
  python scripts/calibrate_stage2_thresholds.py --gt OUT/ref_ground_truth/per_sample.csv --out configs/stage2/thresholds_v2.local.yaml
  python scripts/evaluate_stage2.py --config configs/stage2/ar_gnn.yaml --override "$(cat configs/stage2/thresholds_v2.local.yaml.override)" ...

Nothing is fabricated: the thresholds are data-derived and the file records the quantile + n.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"node_deviation_pct": "node_deviation_pct_max", "density_deviation_pct": "density_deviation_pct_max", "aspl_deviation_pct": "aspl_deviation_pct_max"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="per_sample.csv of the ref_ground_truth evaluation")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--out", default="configs/stage2/thresholds_v2.local.yaml")
    a = ap.parse_args()
    df = pd.read_csv(a.gt)
    th = {}
    for col, key in METRICS.items():
        if col in df:
            th[key] = float(np.ceil(np.nanquantile(df[col].astype(float), a.quantile)))
    th["edge_accuracy_pct_min"] = 70.0  # prototype preservation is a hard invariant; keep
    th["inference_time_s_max"] = 60.0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "# data-derived Stage-2 outline thresholds (quantile %.2f of the ground-truth row, n=%d)\n" % (a.quantile, len(df))
    body += "stage2:\n  evaluator:\n    name: topology_spec\n    params:\n      thresholds: " + json.dumps(th) + "\n"
    out.write_text(body, encoding="utf-8")
    # one-line --override form for shell scripts
    ov = " ".join(f"stage2.evaluator.params.thresholds.{k}={v}" for k, v in th.items())
    Path(str(out) + ".override").write_text(ov + "\n", encoding="utf-8")
    print(json.dumps(th, indent=1))
    print(f"written: {out}  (+ .override)")


if __name__ == "__main__":
    main()
