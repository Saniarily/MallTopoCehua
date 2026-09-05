"""Topology generators (kind ``generator``): rule / search baselines and the learned AR-GNN."""

from mall_space_planner.stage2.generators.rule_expander import RuleBasedExpander
from mall_space_planner.stage2.generators.search_expander import SearchExpander

try:  # optional dependency (torch)
    from mall_space_planner.stage2.generators.ar_gnn import ARGNNExpander
except ImportError:  # pragma: no cover
    ARGNNExpander = None  # type: ignore[assignment,misc]

__all__ = ["RuleBasedExpander", "SearchExpander", "ARGNNExpander"]
