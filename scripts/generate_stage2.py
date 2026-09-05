#!/usr/bin/env python3
"""Run the Stage-2 rule baseline on a prototype from the processed DB and export JSON/GeoJSON/SVG."""
from __future__ import annotations
import json
from _common import ROOT, base_parser
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.geometry.export import layout_to_geojson, layout_to_json, layout_to_svg
from mall_space_planner.visualization.render import render_layout_png
from mall_space_planner.schemas import ConstraintSet, SiteBoundary
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.stage2.pipelines.generate import Stage2Pipeline
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

def main() -> None:
    p = base_parser("Generate stage 2"); p.add_argument("--processed-dir", default="data/processed/synthetic"); p.add_argument("--prototype-id", default=None)
    p.add_argument("--target-nodes", type=int, default=None); p.add_argument("--target-shops", type=int, default=None); p.add_argument("--shop-area", type=float, nargs=2, default=None, metavar=("MIN", "MAX")); p.add_argument("--atria", type=int, default=None); p.add_argument("--width", type=float, default=180); p.add_argument("--height", type=float, default=120)
    a = p.parse_args(); setup_logging(a.log_level); cfg = resolve_config(a.config, a.override); paths = ProjectPaths(root=ROOT)
    db = CaseDatabase.load(paths.resolve(a.processed_dir)); fid = a.prototype_id or next(iter(db.graphs))
    proto = db.get_case(fid).prototype; n_t = a.target_nodes or int(round(proto.graph.num_nodes * 1.6))
    req = GenerationRequest(prototype=proto, boundary=SiteBoundary.rectangle(a.width, a.height), constraints=ConstraintSet(target_num_nodes=n_t, target_num_shops=a.target_shops, shop_area_min=a.shop_area[0] if a.shop_area else None, shop_area_max=a.shop_area[1] if a.shop_area else None, num_atria=a.atria), n_candidates=int(cfg.get("n_candidates", 3)), seed=int(cfg.get("seed", 0)))
    results = Stage2Pipeline(cfg).run(req); out = paths.resolve(cfg.get("output_dir", "outputs/generated_layouts")) / fid; out.mkdir(parents=True, exist_ok=True)
    for i, (lay, res) in enumerate(results):
        layout_to_json(lay, out / f"cand{i}.json"); layout_to_geojson(lay, out / f"cand{i}.geojson"); layout_to_svg(lay, out / f"cand{i}.svg"); render_layout_png(lay, out / f"cand{i}.png")
        keys = ("topo_node_deviation_pct", "topo_edge_accuracy_pct", "topo_density_deviation_pct", "topo_aspl_deviation_pct", "geom_n_shops", "geom_inside_area_ratio", "geom_shop_overlap_rate", "geom_shop_reachable_rate", "geom_constraint_satisfaction_rate", "generation_seconds")
        print(f"cand{i}: pass={res.overall_pass} " + json.dumps({k: (round(res.metrics[k], 3) if isinstance(res.metrics.get(k), float) else res.metrics.get(k)) for k in keys}))
        failed = [k for k, v in res.passed.items() if not v]
        if failed: print(f"   failed checks: {failed}")
    print(f"exports: {out}")

if __name__ == "__main__":
    main()
