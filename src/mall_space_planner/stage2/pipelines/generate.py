"""Compose Stage-2 components from config and produce N candidate layouts with evaluations."""

from __future__ import annotations

import time
from typing import Any

from mall_space_planner.evaluation.stage2_eval import TopologySpecEvaluator
from mall_space_planner.registry import build
from mall_space_planner.schemas import EvaluationResult, GeneratedLayout
from mall_space_planner.stage2.base import GenerationRequest


class Stage2Pipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        s2 = config.get("stage2", {})
        self.generator = build("generator", s2.get("generator", {"name": "rule_expander"}))
        self.decoder = build("geometry_decoder", s2.get("geometry_decoder", {"name": "skeleton_embed"}))
        ev = s2.get("evaluator", {"name": "topology_spec"})
        self.evaluator: TopologySpecEvaluator = build("evaluator", ev)
        self.config = config

    def run(self, request: GenerationRequest) -> list[tuple[GeneratedLayout, EvaluationResult]]:
        out = []
        n_target = request.constraints.target_num_nodes or int(round(request.prototype.graph.num_nodes * 1.5))
        for i in range(request.n_candidates):
            seed = request.seed + i
            t0 = time.perf_counter()
            topo = self.generator.generate(request, seed)
            layout = self.decoder.decode(topo, request, seed)
            dt = time.perf_counter() - t0
            layout.generator_name = getattr(self.generator, "registry_name", type(self.generator).__name__)
            res = self.evaluator.evaluate(request.prototype.graph, topo, n_target, inference_time_s=dt)
            layout.diagnostics.update({"evaluation": res.model_dump()})
            out.append((layout, res))
        out.sort(key=lambda t: (not t[1].overall_pass, t[1].metrics.get("node_deviation_pct") or 0))
        return out
