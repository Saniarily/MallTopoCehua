"""Experiment orchestration: single runs, multi-seed ablations, aggregation and paper tables."""

from mall_space_planner.experiments.runner import run_stage1_experiment
from mall_space_planner.experiments.aggregate import aggregate_runs, to_markdown_table, plot_metric_bars

__all__ = ["run_stage1_experiment", "aggregate_runs", "to_markdown_table", "plot_metric_bars"]
