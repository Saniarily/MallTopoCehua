"""Stage-2 evaluators.

:class:`TopologySpecEvaluator` implements the acceptance protocol of the attached test
outline (5 metrics with thresholds). Thresholds are injected from config so that the
"outline" values (30 / 70 / 40 / 35 / 60) and the "record table" values (20 % node
deviation) can both be expressed without code changes.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from mall_space_planner.registry import register
from mall_space_planner.schemas import EvaluationResult, TopologyGraph
from mall_space_planner.topology.convert import to_networkx
from mall_space_planner.topology.metrics import (
    aspl_deviation,
    compute_topology_metrics,
    density_deviation,
    edge_accuracy,
    node_deviation,
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "node_deviation_pct_max": 30.0,
    "edge_accuracy_pct_min": 70.0,
    "density_deviation_pct_max": 40.0,
    "aspl_deviation_pct_max": 35.0,
    "inference_time_s_max": 60.0,
}


@register("evaluator", "topology_spec")
class TopologySpecEvaluator:
    """Evaluate a generated topology against the skeleton and target scale."""

    def __init__(self, thresholds: dict[str, float] | None = None, require_connected: bool = False) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.require_connected = require_connected

    def evaluate(
        self,
        skeleton: TopologyGraph,
        generated: TopologyGraph,
        n_target: int,
        inference_time_s: float | None = None,
        target: TopologyGraph | None = None,
    ) -> EvaluationResult:
        t = self.thresholds
        metrics: dict[str, float | int | None] = {
            "node_deviation_pct": node_deviation(generated, n_target),
            "edge_accuracy_pct": edge_accuracy(skeleton, generated),
            "density_deviation_pct": density_deviation(skeleton, generated, n_target),
            "aspl_deviation_pct": aspl_deviation(skeleton, generated, n_target),
            "inference_time_s": inference_time_s,
            "n_nodes": generated.num_nodes,
            "n_edges": generated.num_edges,
            "n_components": nx.number_connected_components(to_networkx(generated)) if generated.num_nodes else 0,
        }
        passed = {
            "node_deviation": metrics["node_deviation_pct"] <= t["node_deviation_pct_max"],
            "edge_accuracy": metrics["edge_accuracy_pct"] >= t["edge_accuracy_pct_min"],
            "density_deviation": metrics["density_deviation_pct"] <= t["density_deviation_pct_max"],
            "aspl_deviation": metrics["aspl_deviation_pct"] <= t["aspl_deviation_pct_max"],
        }
        if inference_time_s is not None:
            passed["inference_time"] = inference_time_s <= t["inference_time_s_max"]
        if self.require_connected:
            passed["connected"] = metrics["n_components"] == 1

        details: dict[str, Any] = {"thresholds": t, "generated_metrics": compute_topology_metrics(generated).model_dump()}
        if target is not None:
            # Optional comparison against the ground-truth expansion (when available)
            tm = compute_topology_metrics(target)
            details["target_metrics"] = tm.model_dump()
            metrics["edge_count_error"] = abs(generated.num_edges - target.num_edges)
            metrics["node_count_error"] = abs(generated.num_nodes - target.num_nodes)
            metrics["target_edge_recall_pct"] = edge_accuracy(target, generated)
            metrics["target_edge_precision_pct"] = edge_accuracy(generated, target)
        return EvaluationResult(evaluator="topology_spec", metrics=metrics, passed=passed, overall_pass=all(passed.values()), details=details)
