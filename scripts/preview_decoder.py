#!/usr/bin/env python3
"""Quick visual check of a geometry decoder on real / synthetic topologies.

  python scripts/preview_decoder.py --decoder planar_corridor --corpus /tmp/corpus_v2.jsonl --out /tmp/decoder_preview.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MP  # noqa: E402

from mall_space_planner.data.corpus_builder import load_any_corpus  # noqa: E402
from mall_space_planner.evaluation.geometry_eval import GeometryEvaluator  # noqa: E402
from mall_space_planner.registry import build  # noqa: E402
from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype  # noqa: E402
from mall_space_planner.stage2.base import GenerationRequest  # noqa: E402
from mall_space_planner.stage2.repair.basic import BasicRepairer  # noqa: E402

KINDS = {"shop": "#CFE8FF", "anchor": "#DCD6F7", "corridor": "#F0C987", "atrium": "#B5E7A0"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="planar_corridor")
    ap.add_argument("--corpus", default="data/samples/synthetic/sharegpt_sample.json")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--site", default="180x120")
    ap.add_argument("--out", default="/tmp/decoder_preview.png")
    ap.add_argument("--skeleton", action="store_true", help="decode skeletons instead of targets")
    a = ap.parse_args()
    W, H = (float(x) for x in a.site.lower().split("x"))
    samples = load_any_corpus(a.corpus)[: a.n]
    dec = build("geometry_decoder", {"name": a.decoder})
    ge, rep = GeometryEvaluator(), BasicRepairer()
    cols = min(3, len(samples))
    rows = (len(samples) + cols - 1) // cols
    f, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 3.6 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), samples, strict=False):
        g = s.skeleton if a.skeleton else s.target
        req = GenerationRequest(prototype=TopologyPrototype(prototype_id=s.sample_id, graph=g), boundary=SiteBoundary.rectangle(W, H), constraints=ConstraintSet(target_num_nodes=g.num_nodes, shop_area_min=60, shop_area_max=300), seed=0)
        L = rep.repair(dec.decode(g, req, 0), req)
        ax.add_patch(MP(L.boundary.exterior, closed=True, fc="#fafafa", ec="k"))
        for u in L.units:
            if u.polygon:
                ax.add_patch(MP(u.polygon, closed=True, fc=KINDS.get(u.kind, "#ddd"), ec="#666", lw=0.4))
        pos = L.skeleton_positions
        for u, v in L.topology.edges():
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#D9480F", lw=0.9)
        r = ge.evaluate(L, req.constraints)
        d = L.diagnostics
        ax.set_title(f"{s.sample_id}: shops={r.metrics['n_shops']} reach={r.metrics['shop_reachable_rate']:.2f} atria={d.get('n_loop_atria', '-')} pass={r.overall_pass}", fontsize=8)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
    for ax in axes.ravel()[len(samples):]:
        ax.axis("off")
    f.tight_layout()
    f.savefig(a.out, dpi=90)
    print(a.out)


if __name__ == "__main__":
    main()
