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
    # labelled Procrustes is not a reliable criterion on this elongated floor (random placements score ≈ 0.2·diagonal
    # too, see evaluate.py); only require that it is computed and finite
    assert np.isfinite(ev["procrustes_rmse_m"]) and np.isfinite(ev["procrustes_rmse_rand_m"])
    assert abs(ev["outer_facade_dist_m"] - ev["gt_outer_facade_dist_m"]) < 8.0
    # no more sharp wedges than the real network has (real: 13% of node angles < 60°)
    assert ev["sharp_angle_rate"] < 0.25
    r, prm = procrustes(np.array([pos[k] for k in g.nodes]), np.array([gt[k] for k in g.nodes]))
    assert r == pytest.approx(ev["procrustes_rmse_m"]) and prm["scale"] > 0


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
    assert all(u.kind in {"corridor", "atrium", "entrance", "vertical_core"} for u in layout.units)


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


def test_renovation_mode_anchors_skeleton_and_regrows():
    """Renovation: same outline, skeleton nodes anchored at their real positions, secondary network regrown."""
    from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths
    from mall_space_planner.stage3.renovate import build_generator, renovate_floor

    fix = Path("tests/fixtures/graph_csv")
    ds = Stage3Dataset(Stage3Paths(graph_dir=fix))
    gen, _ = build_generator(None)
    r = renovate_floor("B000A0E928_1", fix, ds, gen, CorridorFitter(FitParams(n_restarts=2, iters=40)), RenderParams(), seed=0, n_candidates=2)
    assert r["after"].num_nodes == r["before"].num_nodes
    # anchored nodes (existing entrances) did not move; skeleton nodes were re-laid-out
    assert r["anchored"] and len(r["entrance_points"]) >= 2
    for v in r["anchored"]:
        assert np.allclose(r["res"].positions[v], r["gt"][v], atol=1e-6)
    assert r["res"].diagnostics["init_mode"] == "outline"
    assert r["row"]["after_crossings"] == 0
    assert r["row"]["after_inside_ratio"] == 1.0
    assert r["plan"].diagnostics["n_entrances"] >= 2
    assert r["plan"].diagnostics["main_width_m"] > r["plan"].diagnostics["secondary_width_m"]
    assert all(k in r["ind_a"] for k in ("num_cycles", "avg_shortest_path", "closeness_mean", "n_dead_ends"))


def test_nonconvex_outline_keeps_corridors_inside():
    """L-shaped outline: no corridor may cut across the re-entrant corner (nodes inside is not enough)."""
    import networkx as nx
    import numpy as np

    from mall_space_planner.schemas import TopologyGraph
    from mall_space_planner.stage3 import CorridorFitter, FitParams
    from mall_space_planner.stage3.fit import edges_outside
    from mall_space_planner.stage3.outline import outline_from_points

    outline = outline_from_points([(0, 0), (140, 0), (140, 90), (70, 90), (70, 55), (0, 55)])
    g = nx.Graph()
    ring = [f"S{i}" for i in range(8)]
    g.add_edges_from(zip(ring, ring[1:] + ring[:1]))
    for i in range(10):  # branches / chords crossing the notch when laid out naively
        g.add_edge(f"N{i}", ring[i % 8]); g.add_edge(f"N{i}", ring[(i + 3) % 8])
    topo = TopologyGraph(adjacency={v: sorted(g.neighbors(v)) for v in g.nodes})
    res = CorridorFitter(FitParams(n_restarts=2, iters=40)).fit(topo, outline, seed=0, skeleton_nodes=set(ring))
    pos = {k: np.asarray(v) for k, v in res.positions.items()}
    assert res.diagnostics["edges_outside"] == 0
    assert edges_outside(g, pos, outline.polygon) == 0


def test_align_outline_to_csv_recovers_shift_and_scale(tmp_path):
    """Mask PNG shifted / rescaled against the graph-CSV pixel frame: the alignment must put the M nodes back inside."""
    import numpy as np
    from PIL import Image, ImageDraw
    from shapely.affinity import scale, translate
    from shapely.geometry import Point, Polygon

    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.stage3.outline import align_outline_to_csv, m_positions_from_total_csv, outline_from_mask, outline_from_total_csv
    from mall_space_planner.topology.convert import to_networkx

    tot = Path("tests/fixtures/graph_csv/B000A0E928_1_total.csv")
    base = outline_from_total_csv(tot)
    poly_px = Polygon(base.m_to_px(np.array(base.polygon.exterior.coords)))
    g = to_networkx(load_target_csv("tests/fixtures/graph_csv/B000A0E928_1_M.csv"))

    def inside_rate(o) -> float:  # noqa: ANN001
        gt = m_positions_from_total_csv(tot, o)
        return float(np.mean([o.polygon.buffer(1.0).covers(Point(gt[v])) for v in g.nodes if v in gt]))

    for i, (shift, sc) in enumerate([((-30, -40), 1.0), ((0, 0), 1.15), ((-25, -35), 0.9)]):
        p = scale(translate(poly_px, *shift), sc, sc, origin=(0, 0))
        _, _, maxx, maxy = p.bounds
        img = Image.new("L", (int(maxx) + 10, int(maxy) + 10), 0)
        ImageDraw.Draw(img).polygon([(x, y) for x, y in p.exterior.coords], fill=255)
        mp = tmp_path / f"m{i}.png"; img.save(mp)
        o = outline_from_mask(mp)
        before = inside_rate(o)
        o = align_outline_to_csv(o, tot)
        assert inside_rate(o) >= 0.98 and inside_rate(o) >= before
        assert o.extra["csv_align"]["mode"] != "identity"
    # already aligned mask: identity kept
    img = Image.new("L", (int(poly_px.bounds[2]) + 10, int(poly_px.bounds[3]) + 10), 0)
    ImageDraw.Draw(img).polygon([(x, y) for x, y in poly_px.exterior.coords], fill=255)
    mp = tmp_path / "same.png"; img.save(mp)
    o = align_outline_to_csv(outline_from_mask(mp), tot)
    assert o.extra["csv_align"]["mode"] == "identity"


def test_dataset_outline_falls_back_when_mask_is_unrelated(tmp_path):
    """Mask PNG of a different region: < 90 % key points inside after alignment -> outline rebuilt from *_total.csv."""
    import shutil

    from PIL import Image, ImageDraw

    from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths
    from mall_space_planner.stage3.outline import nodes_inside_rate

    gd = tmp_path / "g"; gd.mkdir()
    for f in Path("tests/fixtures/graph_csv").glob("B000A0E928_1*"):
        shutil.copy(f, gd / f.name)
    md = tmp_path / "mask"; md.mkdir()
    img = Image.new("L", (400, 300), 0); ImageDraw.Draw(img).rectangle([40, 40, 360, 260], fill=255)  # wide box, not this floor
    img.save(md / "B000A0E928_1_0.png")
    ds = Stage3Dataset(Stage3Paths(graph_dir=gd, outline_mask_dir=md))
    o = ds.outline("B000A0E928_1", 0)
    assert o.extra.get("outline_fallback", {}).get("from") == "total_csv"
    assert nodes_inside_rate(o, gd / "B000A0E928_1_total.csv") >= 0.98


def test_outline_falls_back_to_csv_when_mask_is_unrelated(tmp_path):
    """A mask from another region / crop puts most key points outside even after alignment -> CSV-polygon outline."""
    from PIL import Image, ImageDraw

    from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths
    from mall_space_planner.stage3.outline import nodes_inside_rate

    md = tmp_path / "outer_mask"; md.mkdir()
    img = Image.new("L", (420, 320), 0); ImageDraw.Draw(img).rectangle([10, 10, 400, 300], fill=255); img.save(md / "B000A0E928_1_0.png")
    ds = Stage3Dataset(Stage3Paths(graph_dir=Path("tests/fixtures/graph_csv"), outline_mask_dir=md))
    o = ds.outline("B000A0E928_1", 0)
    assert o.extra.get("outline_fallback", {}).get("from") == "total_csv"
    assert nodes_inside_rate(o, Path("tests/fixtures/graph_csv/B000A0E928_1_total.csv")) >= 0.98


def test_read_csv_robust_one_row(tmp_path):
    from mall_space_planner.data.corpus_builder import load_target_csv

    p = tmp_path / "x_M.csv"; p.write_text("Source,Target,Shared_L_Count,Shared_L_Nodes\nM001,M002,1,[]\n")
    assert load_target_csv(p).num_nodes == 2
