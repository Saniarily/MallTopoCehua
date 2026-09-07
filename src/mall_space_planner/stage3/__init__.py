"""Stage 3 – corridor adaptation: fit an M key-point network into a (possibly new / hand-drawn) outline and
render it as a complete corridor system (widths, hierarchy, entrances), leaving the rest to the designer.

Modules
-------
outline   : outline polygons from mask PNGs, ``*_total.csv`` unit polygons, or hand-drawn point lists
skeleton  : inset region + medial-axis (centre-line) graph of the outline
fit       : ``CorridorFitter`` – place the M nodes inside the outline (planar, straight, outline-aware)
render    : corridor polygons with main/secondary widths, loop atria, entrances to the façade
evaluate  : geometric quality metrics + Procrustes agreement with the real drawing when available
"""

from mall_space_planner.stage3.fit import CorridorFitter, FitParams, FitResult
from mall_space_planner.stage3.outline import Outline, outline_from_mask, outline_from_points, outline_from_total_csv
from mall_space_planner.stage3.render import CorridorPlan, RenderParams, render_corridors

__all__ = [
    "CorridorFitter", "CorridorPlan", "FitParams", "FitResult", "Outline", "RenderParams",
    "outline_from_mask", "outline_from_points", "outline_from_total_csv", "render_corridors",
]
