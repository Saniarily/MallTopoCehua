import json
import pytest
from mall_space_planner.api.service import PlanningService
from mall_space_planner.schemas import ConstraintSet, PlanningCondition, SiteBoundary
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline

S1 = {"seed": 1, "stage1": {"hard_filter": {"min_candidates": 3}, "retriever": {"name": "knn"}, "recall_top_n": 50, "explainer": {"name": "template"}, "counterfactuals": {"enabled": False}}, "eval": {"min_candidates": 3}}
S2 = {"stage2": {"generator": {"name": "search_expander", "params": {"n_trials": 4}}, "geometry_decoder": {"name": "corridor_partition"}, "repairer": {"name": "basic"}}}
Q = PlanningCondition(city_cluster=2, people=8000, GDP_2023=50000, PCDI_2023=55000, TP_2023=5000, mall_area_count=90, nearest_distance_km=0.3, count_1km=8, count_2km=20, total_area=150000, Tx=25)

@pytest.mark.parametrize("ranker", ["lgbm_lambdarank", "lgbm_regressor", "mlp"])
def test_new_rankers_fit_and_rank(synthetic_db, ranker):
    pytest.importorskip("lightgbm") if ranker.startswith("lgbm") else None
    params = {"num_rounds": 30, "candidates_per_query": 8, "early_stopping_rounds": 0} if ranker.startswith("lgbm") else {"pairs_per_query": 5, "max_iter": 50}
    cfg = {**S1, "stage1": {**S1["stage1"], "ranker": {"name": ranker, "params": params}}}
    pipe = Stage1Pipeline(cfg, synthetic_db).fit(); recs = pipe.recommend(Q, top_k=5)
    assert recs and recs[0].explanation and pipe.ranker.feature_importance()

def test_end_to_end_service(synthetic_db, tmp_path):
    cfg = {**S1, "stage1": {**S1["stage1"], "ranker": {"name": "random_forest", "params": {"n_estimators": 20, "pairs_per_query": 5}}}}
    svc = PlanningService(synthetic_db, cfg, S2)
    recs = svc.recommend(Q, top_k=3, with_counterfactuals=False); assert recs
    results = svc.generate(recs[0].prototype_id, SiteBoundary.rectangle(160, 100), ConstraintSet(target_num_nodes=30, target_num_shops=30, shop_area_min=50, shop_area_max=300), n_candidates=2)
    assert len(results) == 2 and all(r.metrics["topo_edge_accuracy_pct"] == 100 for _, r in results)
    files = svc.export(results[0][0], tmp_path, stem="c0"); assert {"json", "geojson", "svg", "png"} <= set(files) and all(p.exists() for p in files.values())
    geo = json.loads(files["geojson"].read_text()); assert any(f["properties"]["kind"] == "shop" for f in geo["features"])
