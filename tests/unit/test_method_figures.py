"""Smoke test: the M-series method-case figures build on the sandbox stand-ins (synthetic DB + fixture floor)."""
from pathlib import Path

import pytest


@pytest.mark.timeout(600)
def test_method_figures_cheap_subset(tmp_path: Path) -> None:
    from mall_space_planner.reporting import figures_method as fm
    from mall_space_planner.reporting.style import apply_style

    apply_style()
    ctx = fm.build_ctx(Path("data/results_snapshot"))
    for fn in (fm.m04_stage2_method_comparison, fm.m07_stage2_growth_steps, fm.m12_data_levels):
        paths = fn(ctx, tmp_path)
        if not paths:  # inputs absent on this machine
            continue
        assert {p.suffix for p in paths} >= {".png", ".svg"}
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)
