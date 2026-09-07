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


def test_stage3_dataset_csv_and_outline_loader(tmp_path) -> None:  # noqa: ANN001
    import yaml

    from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths

    cfg = {"dataset": {"params": {"graph_dir": str(FIX)}}, "stage3": {"dataset_csv": str(FIX / "dataset_0_sample.csv"), "m_per_px": 0.5, "outline_close_px": 20}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ds = Stage3Dataset(Stage3Paths.from_config(cfg))
    recs = ds.regions("B000A0E928_1")
    assert len(recs) == 1 and recs[0].region == 0 and recs[0].area_total_px == 67568.0
    assert abs(recs[0].corridor_ratio - 14938 / 67568) < 1e-9
    st = ds.corridor_ratio_stats()
    assert st["n"] > 0 and 0.05 < st["median"] < 0.4
    o = ds.outline("B000A0E928_1")  # no mask dir configured -> *_total.csv polygons
    assert o.polygon.is_valid and o.extra["real_corridor_ratio"] > 0 and o.scale_source == "default"
    # pixel scale from a mall-level m² area spread over its floors
    mpp, src = ds.m_per_px_for("B000A0E928_1", mall_area_m2=3 * 67568 * 0.25, n_floors=3)
    assert src == "main_table_area" and abs(mpp - 0.5) < 1e-9
    sims = ds.similar_outlines(67568 * 0.25, k=3, exclude_mall="B000A0E928", require_source=False)
    assert sims and all(not s.startswith("B000A0E928") for s in sims)
    # with the default require_source=True only floors whose mask/total.csv exists are returned: none for other malls here,
    # the fixture floor itself when not excluded
    assert ds.similar_outlines(67568 * 0.25, k=3, exclude_mall="B000A0E928") == []
    assert ds.similar_outlines(67568 * 0.25, k=1) == ["B000A0E928_1"]


def test_outline_from_mask_png(tmp_path) -> None:  # noqa: ANN001
    from PIL import Image

    from mall_space_planner.stage3.outline import outline_from_mask

    # L-shaped white region on black, like outer_mask/*.png (255 inside)
    img = np.zeros((200, 300), np.uint8)
    img[40:160, 30:270] = 255
    img[100:160, 150:270] = 0
    p = tmp_path / "X_1_0.png"
    Image.fromarray(img).save(p)
    o = outline_from_mask(p, area_m2=None)
    assert o.polygon.is_valid and o.scale_source == "default"
    # area in px ~ 240*120 - 120*60 = 21600 -> m² at 0.5 m/px
    assert abs(o.area - 21600 * 0.25) / (21600 * 0.25) < 0.03
    # inverted convention (0 inside) gives the same shape
    Image.fromarray(255 - img).save(p)
    o2 = outline_from_mask(p)
    assert abs(o2.area - o.area) / o.area < 0.03
