import pytest
from shapely.geometry import Polygon
from mall_space_planner.evaluation.geometry_eval import GeometryEvaluator
from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyGraph, TopologyPrototype
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.stage2.decoders.corridor_partition import CorridorPartitionDecoder
from mall_space_planner.stage2.generators.rule_expander import RuleBasedExpander
from mall_space_planner.stage2.generators.search_expander import SearchExpander
from mall_space_planner.stage2.repair.basic import BasicRepairer

def _req(n_target=20, boundary=None):
    sk = TopologyGraph(adjacency={"A": ["B"], "B": ["C"], "C": ["D"], "D": ["E"], "E": []}, positions={"A": (0, 0), "B": (1, 0), "C": (2, 0), "D": (3, 0), "E": (4, 0)})
    proto = TopologyPrototype(prototype_id="p", graph=sk)
    return GenerationRequest(prototype=proto, boundary=boundary or SiteBoundary.rectangle(120, 80), constraints=ConstraintSet(target_num_nodes=n_target, shop_area_min=40, shop_area_max=300, num_atria=1), seed=1)

@pytest.mark.parametrize("gen", [RuleBasedExpander(), SearchExpander(n_trials=4)])
def test_generators_preserve_skeleton_and_hit_target(gen):
    req = _req(20); g = gen.generate(req, seed=3)
    assert g.num_nodes == 20 and set(req.prototype.graph.edges()) <= set(g.edges())

def test_partition_no_overlap_inside_boundary_and_repair():
    req = _req(16); topo = RuleBasedExpander().generate(req, 0)
    lay = CorridorPartitionDecoder().decode(topo, req, 0)
    shops = [u for u in lay.units if u.kind == "shop"]; assert shops and any(u.kind == "corridor" for u in lay.units)
    res = GeometryEvaluator().evaluate(lay, req.constraints)
    assert res.metrics["shop_overlap_rate"] < 0.01 and res.metrics["inside_area_ratio"] > 0.98 and res.metrics["invalid_polygon_rate"] == 0
    rep = BasicRepairer().repair(lay, req)
    assert all((u.area or 0) >= 40 for u in rep.units if u.kind == "shop") and rep.diagnostics["n_shops"] <= len(shops)

def test_lshape_boundary_supported():
    b = SiteBoundary(exterior=[(0, 0), (150, 0), (150, 60), (80, 60), (80, 110), (0, 110)])
    req = _req(14, boundary=b); topo = RuleBasedExpander().generate(req, 0)
    lay = BasicRepairer().repair(CorridorPartitionDecoder().decode(topo, req, 0), req)
    poly = Polygon(b.exterior)
    assert all(poly.buffer(1e-6).contains(Polygon(u.polygon)) for u in lay.units if u.polygon)
    assert GeometryEvaluator().evaluate(lay, req.constraints).passed["inside_boundary"]
