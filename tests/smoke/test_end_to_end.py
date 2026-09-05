import pytest
from mall_space_planner.evaluation.stage1_eval import evaluate_stage1
from mall_space_planner.registry import available
from mall_space_planner.schemas import ConstraintSet, PlanningCondition, SiteBoundary
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.stage2.pipelines.generate import Stage2Pipeline
from mall_space_planner.geometry.export import layout_to_geojson, layout_to_svg

CFG = {"seed": 1, "stage1": {"hard_filter": {"min_candidates": 3}, "retriever": {"name": "knn"}, "recall_top_n": 50,
       "ranker": {"name": "random_forest", "params": {"n_estimators": 20, "pairs_per_query": 5}}, "explainer": {"name": "template"},
       "counterfactuals": {"enabled": True, "deltas": {"total_area": 0.3}}}, "eval": {}}

@pytest.mark.smoke
def test_registry_lists_components():
    av = available(); assert "random_forest" in av["ranker"] and "rule_expander" in av["generator"]

@pytest.mark.smoke
@pytest.mark.parametrize("ranker", ["weighted_rule", "random_forest", "ridge"])
def test_stage1_switch_models(synthetic_db, ranker, tmp_path):
    cfg = {**CFG, "stage1": {**CFG["stage1"], "ranker": {"name": ranker, "params": {} if ranker == "weighted_rule" else {"pairs_per_query": 5}}}}
    pipe = Stage1Pipeline(cfg, synthetic_db).fit()
    q = PlanningCondition(city_cluster=2, people=8000, GDP_2023=50000, PCDI_2023=55000, TP_2023=5000, mall_area_count=90, nearest_distance_km=0.3, count_1km=8, count_2km=20, total_area=150000, Tx=25)
    recs = pipe.recommend(q, top_k=5)
    assert 1 <= len(recs) <= 5 and recs[0].explanation is not None and recs[0].confidence is not None
    q2 = q.model_copy(update={"total_area": 600000}); recs2 = pipe.recommend(q2, top_k=5, with_explanations=False)
    assert [r.prototype_id for r in recs] != [r.prototype_id for r in recs2] or len(recs2) > 0
    pipe.save(tmp_path / "ck"); loaded = Stage1Pipeline.load(tmp_path / "ck", synthetic_db)
    assert [r.prototype_id for r in loaded.recommend(q, top_k=3, with_explanations=False)] == [r.prototype_id for r in recs[:3]]
    _, agg = evaluate_stage1(pipe, split="test", ks=(5,), min_candidates=3); assert agg["n_queries"] >= 0

@pytest.mark.smoke
def test_stage2_rule_baseline(synthetic_db, tmp_path):
    fid = synthetic_db.cases.floor_id.iloc[0]; proto = synthetic_db.get_case(fid).prototype
    req = GenerationRequest(prototype=proto, boundary=SiteBoundary.rectangle(150, 100), constraints=ConstraintSet(target_num_nodes=proto.graph.num_nodes * 2), n_candidates=2, seed=0)
    results = Stage2Pipeline({"stage2": {}}).run(req)
    assert len(results) == 2
    for lay, res in results:
        assert res.metrics["topo_edge_accuracy_pct"] == 100.0 and res.metrics["topo_node_deviation_pct"] <= 30 and lay.diagnostics["inside_ratio"] == 1.0
    layout_to_svg(results[0][0], tmp_path / "a.svg"); layout_to_geojson(results[0][0], tmp_path / "a.geojson")
    assert (tmp_path / "a.svg").exists() and (tmp_path / "a.geojson").exists()
