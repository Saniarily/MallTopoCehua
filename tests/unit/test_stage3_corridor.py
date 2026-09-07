"""Stage 3: outline extraction, corridor fitting into real and hand-drawn outlines, corridor rendering."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point

from mall_space_planner.data.corpus_builder import load_target_csv
from mall_space_planner.geometry.planar_embed import count_crossings
from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, outline_from_points, outline_from_total_csv, render_corridors
from mall_space_planner.stage3.evaluate import evaluate_fit, procrustes
from mall_space_planner.stage3.outline import m_positions_from_total_csv
from mall_space_planner.stage3.skeleton import dominant_directions, inset_region, medial_axis_graph
from mall_space_planner.topology.convert import to_networkx

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "graph_csv"
TOTAL = FIX / "B000A0E928_1_total.csv"


@pytest.fixture(scope="module")
def real():  # noqa: ANN201
    outline = outline_from_total_csv(TOTAL)
    topo = load_target_csv(FIX / "B000A0E928_1_M_simplified.csv")
    gt = m_positions_from_total_csv(TOTAL, outline)
    return outline, topo, gt


def test_outline_from_total_csv(real) -> None:  # noqa: ANN001
    outline, topo, gt = real
    assert outline.polygon.is_valid and outline.area > 1000
    assert len(gt) == 50  # all M nodes have CenterPoint
    inside = [outline.polygon.contains(Point(p)) for p in gt.values()]
    assert np.mean(inside) > 0.95  # real M nodes lie inside the reconstructed outline
    assert 0 < len(dominant_directions(outline.polygon)) <= 2
    ins = inset_region(outline, 5.0)
    assert 0 < ins.area < outline.area
    g, lines = medial_axis_graph(outline, 5.0, px=1.0)
    assert lines and all(l.length > 0 for l in lines)


def test_fit_real_outline_is_planar_inside_and_close(real) -> None:  # noqa: ANN001
    outline, topo, gt = real
    res = CorridorFitter(FitParams(n_restarts=3, iters=60)).fit(topo, outline, seed=0)
    g = to_networkx(topo)
    pos = {k: np.asarray(v) for k, v in res.positions.items()}
    assert set(pos) == set(g.nodes)
    assert count_crossings(g, pos) == 0
    assert all(outline.polygon.buffer(0.5).contains(Point(p)) for p in pos.values())
    plan = render_corridors(topo, res.positions, outline, res.roles, RenderParams())
    ev = evaluate_fit(topo, res, plan, outline, gt_positions={k: gt[k] for k in g.nodes})
    assert ev["n_entrances"] >= 2 and 0.05 < ev["corridor_ratio"] < 0.4
    assert ev["ortho_deviation_deg"] < 20
    # agreement with the real drawing (label-free): corridors lie where the real corridors are,
    # clearly better than a random placement, and the outer loop sits at a realistic façade distance
    assert ev["chamfer_m"] < 0.85 * ev["chamfer_rand_m"]
    assert ev["procrustes_rmse_m"] < ev["procrustes_rmse_rand_m"]
    assert abs(ev["outer_facade_dist_m"] - ev["gt_outer_facade_dist_m"]) < 8.0
    r, prm = procrustes(np.array([pos[k] for k in g.nodes]), np.array([gt[k] for k in g.nodes]))
    assert r == pytest.approx(ev["procrustes_rmse_m"]) and 0.3 < prm["scale"] < 3.0


def test_fit_hand_drawn_L_outline(real) -> None:  # noqa: ANN001
    _, topo, _ = real
    outline = outline_from_points([(0, 0), (160, 0), (160, 60), (100, 60), (100, 110), (0, 110)])
    res = CorridorFitter(FitParams(n_restarts=2, iters=60)).fit(topo, outline, seed=1)
    g = to_networkx(topo)
    pos = {k: np.asarray(v) for k, v in res.positions.items()}
    assert count_crossings(g, pos) == 0
    assert all(outline.polygon.buffer(0.5).contains(Point(p)) for p in pos.values())
    plan = render_corridors(topo, res.positions, outline, res.roles)
    assert not plan.corridor_union.is_empty and plan.corridor_union.within(outline.polygon.buffer(0.5))
    assert any(u.kind == "entrance" for u in plan.units) and any(u.kind == "corridor" for u in plan.units)
    kinds = {c for c in plan.edge_class.values()}
    assert "main" in kinds


def test_corridor_only_decoder_and_service_api(real) -> None:  # noqa: ANN001
    from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype
    from mall_space_planner.stage2.base import GenerationRequest
    from mall_space_planner.stage2.decoders import CorridorOnlyDecoder

    _, topo, _ = real
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id="x", graph=topo), boundary=SiteBoundary.rectangle(150, 90), constraints=ConstraintSet(target_num_nodes=topo.num_nodes), seed=0)
    layout = CorridorOnlyDecoder(n_restarts=2, iters=40).decode(topo, req, 0)
    assert len(layout.skeleton_positions) == topo.num_nodes
    assert layout.diagnostics["crossings"] == 0 and layout.diagnostics["n_entrances"] >= 1
    assert all(u.kind in {"corridor", "atrium", "entrance"} for u in layout.units)
