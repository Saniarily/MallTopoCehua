#!/usr/bin/env python3
"""Generate the M-series *method case* figures (real-data cases per thesis section; png + pdf + svg, editable text).

  python scripts/make_method_figures.py                                  # all 12 figures -> outputs/thesis/figures_method
  python scripts/make_method_figures.py --only M04 M08 M11               # a subset
  python scripts/make_method_figures.py --out docs/thesis/figures_method # write next to the thesis

On the Mac the figures use the real case DB (data/processed/legacy), corpus v2, the graph CSV dir and the AR-GNN
checkpoints (outputs/checkpoints/stage2/*). In the sandbox the same code runs on the synthetic DB + fixture floor and
the figures carry a red "沙箱示意" watermark. See src/mall_space_planner/reporting/figures_method.py for the list.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from _common import ROOT

import mall_space_planner.reporting.style as st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results_snapshot")
    ap.add_argument("--out", default="outputs/thesis/figures_method")
    ap.add_argument("--style", default="configs/thesis/style.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="figure codes, e.g. M04 M11")
    a = ap.parse_args()
    st._STYLE = None
    style = st.load_style(ROOT / a.style)
    font = st.apply_style(style)
    print(f"font: {font}")
    from mall_space_planner.reporting import figures_method as fm

    results, out = ROOT / a.results, (ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out))
    out.mkdir(parents=True, exist_ok=True)
    ctx = fm.build_ctx(results)
    print(f"case DB: {'real' if ctx.is_real_db else 'synthetic'} | corpus: {ctx.corpus_path} ({'real' if ctx.is_real_corpus else 'stand-in'}, {len(ctx.samples)} test samples) | graphs: {ctx.graph_dir} ({'real' if ctx.is_real_graphs else 'fixture'})")
    index, failed = [], []
    for fn in fm.ALL:
        code = fn.__name__.split("_")[0].upper()
        if a.only and code not in {c.upper() for c in a.only}:
            continue
        t0 = time.time()
        try:
            paths = fn(ctx, out)
            if paths:
                index.append({"code": code, "doc": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "", "files": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths]})
                print(f"[ok] {code} -> {paths[0].name} ({time.time() - t0:.0f}s)")
            else:
                print(f"[skip] {code} (inputs not found)")
        except Exception as exc:  # noqa: BLE001
            failed.append(code)
            print(f"[FAIL] {code}: {exc}")
            traceback.print_exc()
    for n in ctx.notes:
        print(f"[note] {n}")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# M 系列方法案例图目录", ""] + [f"- **{i['code']}** {i['doc']} — `{i['files'][0]}`" for i in index]
    (out / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{len(index)} figures -> {out}" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
