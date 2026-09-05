"""Metric-guided search around the rule expander (non-neural, spec-aware baseline).

For each requested candidate, ``n_trials`` rule expansions are drawn with perturbed
operation probabilities; the trial minimising a weighted sum of the spec deviations
(node / density / ASPL, plus a connectivity penalty) is returned. This is deliberately a
*search* baseline: any learned generator must beat it under the same evaluator to justify
its complexity. Trials are cheap (~ms), so ``n_trials=16`` keeps generation well under
the 60 s budget.
"""

from __future__ import annotations

import numpy as np

from mall_space_planner.registry import register
from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.stage2.base import BaseTopologyGenerator, GenerationRequest
from mall_space_planner.stage2.generators.rule_expander import RuleBasedExpander
from mall_space_planner.topology.convert import to_networkx
from mall_space_planner.topology.metrics import aspl_deviation, density_deviation, node_deviation

import networkx as nx


@register("generator", "search_expander")
class SearchExpander(BaseTopologyGenerator):
    def __init__(self, n_trials: int = 16, w_node: float = 1.0, w_density: float = 1.0, w_aspl: float = 1.5, w_disconnected: float = 100.0, label_style: str = "letters") -> None:
        self.n_trials = n_trials
        self.w = (w_node, w_density, w_aspl, w_disconnected)
        self.label_style = label_style

    def _objective(self, skeleton: TopologyGraph, g: TopologyGraph, n_target: int) -> float:
        wn, wd, wa, wc = self.w
        comps = nx.number_connected_components(to_networkx(g)) if g.num_nodes else 1
        return wn * node_deviation(g, n_target) + wd * density_deviation(skeleton, g, n_target) + wa * aspl_deviation(skeleton, g, n_target) + wc * max(0, comps - 1)

    def generate(self, request: GenerationRequest, seed: int) -> TopologyGraph:
        rng = np.random.RandomState(seed)
        skeleton = request.prototype.graph
        n_target = request.constraints.target_num_nodes or int(round(skeleton.num_nodes * 1.5))
        best, best_obj, trace = None, float("inf"), []
        for t in range(self.n_trials):
            probs = rng.dirichlet([2.0, 2.0, 1.0])  # subdivide / branch / chord
            gen = RuleBasedExpander(op_probs={"subdivide": float(probs[0]), "branch": float(probs[1]), "chord": float(probs[2])}, label_style=self.label_style)
            g = gen.generate(request, seed * 1000 + t)
            obj = self._objective(skeleton, g, n_target)
            trace.append(obj)
            if obj < best_obj:
                best, best_obj = g, obj
        assert best is not None
        best.node_attrs.setdefault("__meta__", {})["search"] = {"n_trials": self.n_trials, "best_objective": best_obj, "trace_min": float(min(trace)), "trace_median": float(np.median(trace))}
        return best
