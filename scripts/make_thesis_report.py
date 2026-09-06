#!/usr/bin/env python3
"""Generate all thesis figures (png+pdf+svg) and a figure index from a results directory.

  python scripts/make_thesis_report.py                                   # results = data/results_snapshot, out = outputs/thesis
  python scripts/make_thesis_report.py --only R05 R09                    # regenerate selected figures
  python scripts/make_thesis_report.py --results data/results_snapshot --style configs/thesis/style.yaml --out outputs/thesis

Edit configs/thesis/style.yaml to change fonts / colours / labels / titles; edit the functions in
src/mall_space_planner/reporting/figures_*.py to change layout. Every figure is a standalone function.
"""
from __future__ import annotations
import argparse, json, traceback
from pathlib import Path
from _common import ROOT
import mall_space_planner.reporting.style as st

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results_snapshot"); ap.add_argument("--out", default="outputs/thesis/figures")
    ap.add_argument("--style", default="configs/thesis/style.yaml"); ap.add_argument("--only", nargs="*", default=None, help="figure codes, e.g. R01 F06")
    a = ap.parse_args()
    st._STYLE = None; style = st.load_style(ROOT / a.style); font = st.apply_style(style)
    print(f"font: {font}")
    from mall_space_planner.reporting import figures_process, figures_stage1, figures_stage2
    results, out = ROOT / a.results, ROOT / a.out
    # copy processed data pointer if present (F02/F03 read cases.csv)
    index, failed = [], []
    for fn in figures_process.ALL + figures_stage1.ALL + figures_stage2.ALL:
        code = fn.__name__.split("_")[0].upper()
        if a.only and code not in {c.upper() for c in a.only}:
            continue
        try:
            paths = fn(results, out)
            if paths:
                index.append({"code": code, "title": st.title_for(code), "files": [str(p.relative_to(ROOT)) for p in paths]}); print(f"[ok] {code} {st.title_for(code)} -> {paths[0].name}")
            else:
                print(f"[skip] {code} (inputs not found)")
        except Exception as exc:  # noqa: BLE001
            failed.append(code); print(f"[FAIL] {code}: {exc}"); traceback.print_exc()
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 图目录", ""] + [f"- **{i['code']}** {i['title']} — `{i['files'][0]}`" for i in index]
    (out / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{len(index)} figures -> {out}" + (f"; failed: {failed}" if failed else ""))

if __name__ == "__main__":
    main()
