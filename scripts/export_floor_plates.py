#!/usr/bin/env python
"""Export every real floor as two figures + a metadata table (thesis appendix; Viewer Hub thumbnail cache).

For each floor ``{mall}_{k}`` whose graph CSVs are reachable (``*_M.csv``, ``*_M_simplified.csv``, ``*_total.csv``):

* ``<out>/plan_topo/<floor>.png``    colour-block plan (``clean_img``) as background + full M network (skeleton edges bold)
* ``<out>/outline_topo/<floor>.png`` outline only + full M network
* ``<out>/thumbs/<floor>.png``       360 px square thumbnail (plan background when available) used by the Viewer Hub
* ``<out>/floors.csv`` / ``floors.json``  per-floor metadata: mall, floor, score, layout type, split, nodes, edges, area m²,
  entrances (façade dead ends), alignment mode/shift, has_plan_png, status

The graph-CSV pixel frame is aligned to the mask (``align_outline_to_csv``) so the network sits inside the outline.
Resumable (skips floors whose three images exist unless ``--force``); ``--workers`` renders in parallel processes.

Usage (Mac):
  python scripts/export_floor_plates.py --config configs/data/legacy.yaml --out outputs/floor_plates
  python scripts/export_floor_plates.py --config configs/data/legacy.yaml --out outputs/floor_plates --floors B000A08791_1 B000A384A3_1 --force
Sandbox smoke test:
  python scripts/export_floor_plates.py --graph-dir tests/fixtures/graph_csv --out /tmp/floor_plates
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mall_space_planner.hub.plates import THUMB_PX, floor_ids_from_graph_dir, render_floor_plates  # noqa: E402
from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths  # noqa: E402
from mall_space_planner.utils.config import resolve_config  # noqa: E402


def _paths(a: argparse.Namespace) -> Stage3Paths:
    if a.config:
        p = Stage3Paths.from_config(resolve_config(a.config, a.override))
    else:
        p = Stage3Paths()
    if a.graph_dir:
        p.graph_dir = Path(a.graph_dir)
    if a.plan_dir:
        p.plan_png_dir = Path(a.plan_dir)
    if a.mask_dir:
        p.outline_mask_dir = Path(a.mask_dir)
    return p


def _paths_from_cfg(cfg: dict) -> Stage3Paths:
    p = Stage3Paths.from_config(resolve_config(cfg["config"], cfg["override"])) if cfg.get("config") else Stage3Paths()
    if cfg.get("graph_dir"):
        p.graph_dir = Path(cfg["graph_dir"])
    if cfg.get("plan_dir"):
        p.plan_png_dir = Path(cfg["plan_dir"])
    if cfg.get("mask_dir"):
        p.outline_mask_dir = Path(cfg["mask_dir"])
    return p


def _one(args: tuple) -> dict:
    fid, cfg, out, force = args
    try:
        ds = Stage3Dataset(_paths_from_cfg(cfg))
        cases = None
        if cfg.get("cases_csv"):
            import pandas as pd

            cases = pd.read_csv(cfg["cases_csv"])
        return render_floor_plates(fid, ds, Path(out), cases=cases, force=force, thumb_px=cfg.get("thumb_px", THUMB_PX), dpi=cfg.get("dpi", 130))
    except Exception as exc:  # noqa: BLE001
        import traceback

        tb = traceback.extract_tb(exc.__traceback__)[-1]
        return {"floor_id": fid, "status": f"error: {type(exc).__name__}: {exc}", "error_at": f"{Path(tb.filename).name}:{tb.lineno}"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="data config with a stage3 block (configs/data/legacy.yaml)")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--graph-dir", default=None); p.add_argument("--plan-dir", default=None); p.add_argument("--mask-dir", default=None)
    p.add_argument("--cases-csv", default=None, help="processed cases.csv for score / layout type (default: from config processed_dir)")
    p.add_argument("--floors", nargs="*", default=None); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="outputs/floor_plates"); p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=1); p.add_argument("--dpi", type=int, default=130); p.add_argument("--thumb-px", type=int, default=THUMB_PX)
    a = p.parse_args()

    paths = _paths(a)
    if paths.graph_dir is None or not paths.graph_dir.exists():
        sys.exit(f"graph dir not found: {paths.graph_dir}")
    floors = a.floors or floor_ids_from_graph_dir(paths.graph_dir)
    if a.limit:
        floors = floors[: a.limit]
    cases_csv = a.cases_csv
    if cases_csv is None and a.config:
        try:
            pd_dir = resolve_config(a.config, a.override).get("processed_dir") or (resolve_config(a.config, a.override).get("data") or {}).get("processed_dir")
            if pd_dir and (ROOT / pd_dir / "cases.csv").exists():
                cases_csv = str(ROOT / pd_dir / "cases.csv")
        except Exception:  # noqa: BLE001
            cases_csv = None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = {"config": a.config, "override": a.override, "graph_dir": str(paths.graph_dir), "plan_dir": str(paths.plan_png_dir) if paths.plan_png_dir else None, "mask_dir": str(paths.outline_mask_dir) if paths.outline_mask_dir else None, "cases_csv": cases_csv, "thumb_px": a.thumb_px, "dpi": a.dpi}
    print(f"{len(floors)} floors -> {out}  (plan_dir={cfg['plan_dir']}, mask_dir={cfg['mask_dir']}, cases={cases_csv})", flush=True)
    t0 = time.time(); rows = []
    jobs = [(fid, cfg, str(out), a.force) for fid in floors]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(_one, j): j[0] for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result(); rows.append(r)
                print(f"[{i}/{len(jobs)}] {r['floor_id']}: {r['status']}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r = _one(j); rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['floor_id']}: {r['status']}  ({time.time() - t0:.0f}s)", flush=True)
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("floor_id")
    # merge with an existing table (resumed runs keep older rows)
    prev = out / "floors.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        df = pd.concat([old[~old["floor_id"].isin(df["floor_id"])], df], ignore_index=True).sort_values("floor_id")
    df.to_csv(prev, index=False)
    (out / "floors.json").write_text(df.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    ok = int((df["status"] == "ok").sum())
    print(f"done: {ok}/{len(df)} ok, {len(df) - ok} skipped/failed, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
