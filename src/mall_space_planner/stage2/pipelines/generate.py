"""Stage-2 pipeline: generator → decoder → repairer → evaluators, producing N ranked candidates.

Candidates are ranked by (all hard checks passed, constraint satisfaction rate,
−node deviation). Results carry both evaluations in ``diagnostics`` for the UI/export.
"""

from __future__ import annotations

import time
from typing import Any

from mall_space_planner.evaluation.geometry_eval import GeometryEvaluator
from mall_space_planner.evaluation.stage2_eval import TopologySpecEvaluator
from mall_space_planner.registry import build
from mall_space_planner.schemas import EvaluationResult, GeneratedLayout
from mall_space_planner.stage2.base import BaseRepairer, GenerationRequest


class Stage2Pipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        s2 = config.get("stage2", {})
        self.generator = build("generator", s2.get("generator", {"name": "rule_expander"}))
        self.decoder = build("geometry_decoder", s2.get("geometry_decoder", {"name": "planar_corridor"}))
        rep = s2.get("repairer", {"name": "basic"})
        self.repairer: BaseRepairer | None = build("repairer", rep) if rep else None
        self.topo_eval: TopologySpecEvaluator = build("evaluator", s2.get("evaluator", {"name": "topology_spec"}))
        self.geom_eval: GeometryEvaluator = build("evaluator", s2.get("geometry_evaluator", {"name": "geometry"}))
        self.config = config

    def run(self, request: GenerationRequest) -> list[tuple[GeneratedLayout, EvaluationResult]]:
        out: list[tuple[GeneratedLayout, EvaluationResult]] = []
        n_target = request.constraints.target_num_nodes or int(round(request.prototype.graph.num_nodes * 1.5))
        for i in range(request.n_candidates):
            seed = request.seed + i
            t0 = time.perf_counter()
            topo = self.generator.generate(request, seed)
            layout = self.decoder.decode(topo, request, seed)
            if self.repairer is not None:
                layout = self.repairer.repair(layout, request)
            dt = time.perf_counter() - t0
            layout.generator_name = getattr(self.generator, "registry_name", type(self.generator).__name__)
            t_res = self.topo_eval.evaluate(request.prototype.graph, layout.topology, n_target, inference_time_s=dt)
            g_res = self.geom_eval.evaluate(layout, request.constraints)
            combined = EvaluationResult(
                evaluator="stage2_combined",
                metrics={**{f"topo_{k}": v for k, v in t_res.metrics.items()}, **{f"geom_{k}": v for k, v in g_res.metrics.items()}, "generation_seconds": dt},
                passed={**{f"topo_{k}": v for k, v in t_res.passed.items()}, **{f"geom_{k}": v for k, v in g_res.passed.items()}},
                overall_pass=bool(t_res.overall_pass and g_res.overall_pass),
                details={"topology": t_res.model_dump(), "geometry": g_res.model_dump()},
            )
            layout.diagnostics.update({"evaluation": combined.model_dump()})
            out.append((layout, combined))
        out.sort(key=lambda t: (not t[1].overall_pass, -(t[1].metrics.get("geom_constraint_satisfaction_rate") or 0), t[1].metrics.get("topo_node_deviation_pct") or 0))
        return out
