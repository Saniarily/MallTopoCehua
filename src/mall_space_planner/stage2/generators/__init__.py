"""Topology generators. Rule-based baseline is always available; learned models are added in Phase 4."""

from mall_space_planner.stage2.generators.rule_expander import RuleBasedExpander

__all__ = ["RuleBasedExpander"]
