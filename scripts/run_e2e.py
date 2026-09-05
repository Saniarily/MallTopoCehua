#!/usr/bin/env python3
"""End-to-end: planning condition JSON → Top-K prototypes → pick → constrained generation → export.

Example:
  python scripts/run_e2e.py --stage1-config configs/stage1/lgbm_lambdarank.yaml \
      --stage2-config configs/stage2/rule_baseline.yaml --condition data/samples/query_example.json \
      --pick 1 --width 180 --height 120 --target-nodes 40 --target-shops 50 --shop-area 60 300
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from _common import ROOT
from mall_space_planner.api.service import PlanningService
from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.schemas import ConstraintSet, PlanningCondition, SiteBoundary
from mall_space_planner.utils import ProjectPaths, resolve_config, setup_logging

def main() -> None:
    p = argparse.ArgumentParser(description="Run the full planning flow")
    p.add_argument("--stage1-config", default="configs/stage1/lgbm_lambdarank.yaml"); p.add_argument("--stage2-config", default="configs/stage2/rule_baseline.yaml")
    p.add_argument("--checkpoint", default=None, help="Stage-1 checkpoint dir (fits in-process if omitted)")
    p.add_argument("--condition", required=True, help="JSON file with PlanningCondition fields"); p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--pick", type=int, default=1, help="1-based rank of the prototype to expand"); p.add_argument("--boundary", default=None, help="JSON file with SiteBoundary; default rectangle")
    p.add_argument("--width", type=float, default=180); p.add_argument("--height", type=float, default=120)
    p.add_argument("--target-nodes", type=int, default=None); p.add_argument("--target-shops", type=int, default=None); p.add_argument("--shop-area", type=float, nargs=2, default=None); p.add_argument("--atria", type=int, default=None)
    p.add_argument("--n-candidates", type=int, default=3); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="outputs/generated_layouts/e2e"); p.add_argument("--override", nargs="*", default=[])
    a = p.parse_args(); setup_logging("INFO"); paths = ProjectPaths(root=ROOT)
    c1 = resolve_config(paths.resolve(a.stage1_config), a.override); c2 = resolve_config(paths.resolve(a.stage2_config))
    db = CaseDatabase.load(paths.resolve(c1["data"]["processed_dir"]))
    svc = PlanningService(db, c1, c2, checkpoint_dir=paths.resolve(a.checkpoint) if a.checkpoint else None)
    cond = PlanningCondition(**json.loads(Path(a.condition).read_text(encoding="utf-8")))
    recs = svc.recommend(cond, top_k=a.top_k)
    out = paths.resolve(a.out); out.mkdir(parents=True, exist_ok=True); svc.recommendations_to_json(recs, out / "recommendations.json")
    print(f"\n=== Stage 1: Top-{len(recs)} prototypes ===")
    for r in recs:
        print(f"#{r.rank:2d} {r.prototype_id:16s} score={r.score:.3f} conf={r.confidence:.2f} quality={r.quality_score} layout={r.layout_type.value if r.layout_type else '-'}")
    if not recs: raise SystemExit("No recommendations (check hard constraints / candidate pool)")
    chosen = recs[min(a.pick, len(recs)) - 1]; print(f"\n{chosen.explanation.recommendation_summary if chosen.explanation else ''}")
    boundary = SiteBoundary(**json.loads(Path(a.boundary).read_text(encoding="utf-8"))) if a.boundary else SiteBoundary.rectangle(a.width, a.height)
    proto = svc.prototype(chosen.prototype_id)
    cons = ConstraintSet(target_num_nodes=a.target_nodes or int(round(proto.graph.num_nodes * 1.6)), target_num_shops=a.target_shops, shop_area_min=a.shop_area[0] if a.shop_area else None, shop_area_max=a.shop_area[1] if a.shop_area else None, num_atria=a.atria, layout_type=chosen.layout_type)
    results = svc.generate(chosen.prototype_id, boundary, cons, n_candidates=a.n_candidates, seed=a.seed)
    print(f"\n=== Stage 2: {len(results)} candidates from prototype {chosen.prototype_id} ({proto.graph.num_nodes} skeleton nodes → target {cons.target_num_nodes}) ===")
    for i, (lay, res) in enumerate(results):
        m = res.metrics; files = svc.export(lay, out / chosen.prototype_id, stem=f"cand{i}")
        print(f"cand{i}: pass={res.overall_pass} nodeDev={m['topo_node_deviation_pct']:.1f}% edgeAcc={m['topo_edge_accuracy_pct']:.0f}% densDev={m['topo_density_deviation_pct']:.1f}% asplDev={m['topo_aspl_deviation_pct']:.1f}% shops={m['geom_n_shops']} inside={m['geom_inside_area_ratio']:.2f} overlap={m['geom_shop_overlap_rate']:.3f} reach={m['geom_shop_reachable_rate']:.2f} satisfy={m['geom_constraint_satisfaction_rate']:.2f} t={m['generation_seconds']:.2f}s")
        failed = [k for k, v in res.passed.items() if not v]
        if failed: print(f"        diagnostics: failed={failed}; repairs={lay.diagnostics.get('repairs', [])[:3]}")
        print(f"        exports: {', '.join(str(v.name) for v in files.values())}")
    print(f"\nall outputs: {out}")

if __name__ == "__main__":
    main()
