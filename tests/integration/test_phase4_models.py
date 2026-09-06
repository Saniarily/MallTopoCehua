"""Phase 4 components: type-conditional quality model, deep residual ranker, AR-GNN generator.

All tests run on the synthetic fixture only (seconds, CPU). They check *mechanics* (fit/predict/save/load/
schema contracts), never accuracy - real-data numbers come from the round-3 scripts on the user's machine.
"""
from __future__ import annotations
import numpy as np
import pytest
from mall_space_planner.api.service import PlanningService
from mall_space_planner.schemas import ConstraintSet, PlanningCondition, SiteBoundary, TopologyPrototype
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline

S1 = {"seed": 1, "stage1": {"hard_filter": {"min_candidates": 3}, "retriever": {"name": "knn"}, "recall_top_n": 50, "explainer": {"name": "template"}, "counterfactuals": {"enabled": False},
                            "type_recommender": {"enabled": True, "n_estimators": 30, "min_samples_leaf": 3, "n_bootstrap": 4}}, "eval": {"min_candidates": 3}}
S2 = {"stage2": {"generator": {"name": "rule_expander"}, "geometry_decoder": {"name": "corridor_partition"}, "repairer": {"name": "basic"}}}
Q = PlanningCondition(city_cluster=2, people=8000, GDP_2023=50000, PCDI_2023=55000, TP_2023=5000, mall_area_count=90, nearest_distance_km=0.3, count_1km=8, count_2km=20, total_area=150000, Tx=25)


# ---------------------------------------------------------------- type recommender: E[score | conditions, type]
def test_type_recommender_ranks_all_types_with_ci(synthetic_db):
    from mall_space_planner.stage1.type_recommender import TreeTypeRecommender, evaluate_type_recommender
    rec = TreeTypeRecommender(n_estimators=30, min_samples_leaf=3, n_bootstrap=4, seed=1).fit(synthetic_db, synthetic_db.split("train"))
    res = rec.recommend(synthetic_db, Q)
    assert res.recommendations, "must return at least one layout type"
    ranks = [r.rank for r in res.recommendations]; assert ranks == list(range(1, len(ranks) + 1))
    exp = [r.expected_score for r in res.recommendations]; assert exp == sorted(exp, reverse=True)
    for r in res.recommendations:
        assert r.ci_low <= r.expected_score <= r.ci_high
        assert 0 <= r.share_in_comparable <= 1 and r.n_comparable_cases >= 0
    assert np.isfinite(res.conditions_only_score)
    ev = evaluate_type_recommender(rec, synthetic_db, "test")
    for k in ("rmse_with_type", "rmse_conditions_only", "spearman_with_type", "spearman_conditions_only", "per_cluster", "best_type_agreement_rate", "policy_uplift"):
        assert k in ev, k


def test_service_type_then_within_type_flow(synthetic_db):
    svc = PlanningService(synthetic_db, {**S1, "stage1": {**S1["stage1"], "ranker": {"name": "ridge", "params": {"pairs_per_query": 3}}}}, S2)
    types = svc.recommend_types(Q); assert types.recommendations
    top = types.recommendations[0].layout_type
    recs = svc.recommend_within_type(Q, top, top_k=3); assert recs
    # every retrieved prototype must be of the user-selected type (hard filter is strict once the type exists)
    assert all(str(getattr(r.layout_type, "value", r.layout_type)) == str(getattr(top, "value", top)) for r in recs), [r.layout_type for r in recs]


# ---------------------------------------------------------------- deep residual Transformer+GNN ranker
@pytest.mark.parametrize("variant", [dict(use_transformer=True, use_gnn=True), dict(use_transformer=False, use_gnn=True), dict(use_transformer=True, use_gnn=False)])
def test_deep_residual_ranker_fits_and_stays_residual(synthetic_db, variant):
    pytest.importorskip("torch")
    params = {"d_model": 16, "epochs": 3, "patience": 2, "ensemble_k": 1, "candidates_per_query": 6, "batch_groups": 4, "device": "cpu", "base_ranker": "ridge", **variant}
    cfg = {**S1, "stage1": {**S1["stage1"], "ranker": {"name": "deep_residual", "params": params}}}
    pipe = Stage1Pipeline(cfg, synthetic_db).fit(); recs = pipe.recommend(Q, top_k=5, with_explanations=False)
    assert len(recs) == 5 and all(np.isfinite(r.score) for r in recs)
    scores = [r.score for r in recs]; assert scores == sorted(scores, reverse=True)
    imp = pipe.ranker.feature_importance(); assert isinstance(imp, dict) and imp


def test_deep_residual_ranker_is_picklable_without_torch_state(synthetic_db, tmp_path):
    pytest.importorskip("torch"); import joblib
    from mall_space_planner.registry import build
    r = build("ranker", {"name": "deep_residual", "params": {"d_model": 16, "epochs": 2, "ensemble_k": 1, "candidates_per_query": 6, "device": "cpu", "base_ranker": "ridge"}})
    assert r.__getstate__()["_model"] is None
    joblib.dump(r, tmp_path / "r.joblib"); assert (tmp_path / "r.joblib").exists()


# ---------------------------------------------------------------- AR-GNN autoregressive expander
def _synthetic_expansion_samples(n: int = 24, seed: int = 0):
    """Build skeleton->target pairs from the synthetic sharegpt sample or, failing that, from stage-2 DB samples."""
    from mall_space_planner.data.sharegpt_adapter import load_sharegpt
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data/samples/synthetic/sharegpt_sample.json"
    if p.exists():
        return load_sharegpt(p, limit=n)
    pytest.skip("synthetic sharegpt sample not present")


def test_ar_gnn_teacher_steps_are_consistent():
    pytest.importorskip("torch")
    from mall_space_planner.stage2.generators.ar_gnn import canonical_order, teacher_steps
    s = _synthetic_expansion_samples(3)[0]
    order = canonical_order(s.skeleton, s.target)
    assert set(order) == set(s.target.nodes) - set(s.skeleton.nodes)
    steps = teacher_steps(s.skeleton, s.target); assert 0 < len(steps) <= len(order)
    st = steps[0]; assert st["valid"] and all(0 <= i < len(st["present"]) for i in st["valid"])
    # label order: the corpus' own creation order (deferred only when no neighbour is present yet)
    assert canonical_order(s.skeleton, s.target, "bfs") and set(canonical_order(s.skeleton, s.target, "bfs")) == set(order)


def test_ar_gnn_fit_generate_save_load_and_evaluate(tmp_path):
    torch = pytest.importorskip("torch")
    from mall_space_planner.evaluation.stage2_eval import TopologySpecEvaluator
    from mall_space_planner.stage2.base import GenerationRequest
    from mall_space_planner.stage2.generators.ar_gnn import ARGNNExpander
    samples = _synthetic_expansion_samples(24)
    gen = ARGNNExpander(d_model=16, n_layers=1, epochs=2, patience=2, ensemble_k=1, device="cpu", seed=1).fit(samples[:20])
    assert gen.history_ and all(np.isfinite(v[-1]) for v in gen.history_.values() if v)
    out = gen.save(tmp_path / "ckpt"); assert (out / "ar_gnn.pt").exists() and (out / "meta.json").exists()
    gen2 = ARGNNExpander(checkpoint=str(out), device="cpu")
    s = samples[-1]; n_t = s.target_num_nodes or s.target.num_nodes
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id=s.sample_id, graph=s.skeleton, layout_type=s.layout_type), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=n_t, layout_type=s.layout_type), seed=3)
    g = gen2.generate(req, seed=3)
    # prototype preservation: all skeleton nodes & edges survive; node count hits the target
    assert set(s.skeleton.nodes) <= set(g.nodes)
    assert set(s.skeleton.edges()) <= set(g.edges())
    assert g.num_nodes == n_t
    r = TopologySpecEvaluator().evaluate(s.skeleton, g, n_t, inference_time_s=0.1, target=s.target)
    assert r.metrics["edge_accuracy_pct"] == 100 and "target_edge_recall_pct" in r.metrics
    # determinism under fixed seed
    g_b = gen2.generate(req, seed=3); assert g_b.edges() == g.edges()


def test_ar_gnn_best_of_reranking_respects_target(tmp_path):
    pytest.importorskip("torch")
    from mall_space_planner.stage2.base import GenerationRequest
    from mall_space_planner.stage2.generators.ar_gnn import ARGNNExpander
    samples = _synthetic_expansion_samples(12)
    gen = ARGNNExpander(d_model=16, n_layers=1, epochs=1, ensemble_k=1, device="cpu", seed=2, best_of=4).fit(samples[:10])
    s = samples[-1]; n_t = s.target_num_nodes or s.target.num_nodes
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id=s.sample_id, graph=s.skeleton, layout_type=s.layout_type), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=n_t, layout_type=s.layout_type), seed=0)
    g = gen.generate(req, seed=0); assert g.num_nodes == n_t


def test_gin_readout_maxpool_matches_scatter_reduce_and_device_policy():
    """MPS lacks scatter_reduce; the dense one-hot max-pool must equal it exactly, and `auto` must never pick MPS."""
    torch = pytest.importorskip("torch")
    from mall_space_planner.stage1.rankers.deep_ranker import select_device
    g = torch.Generator().manual_seed(0); h = torch.randn(37, 8, generator=g); batch = torch.randint(0, 5, (37,), generator=g); batch[:5] = torch.arange(5)
    ref = torch.full((5, 8), -1e9).scatter_reduce(0, batch[:, None].expand_as(h), h, reduce="amax")
    from mall_space_planner.stage1.rankers.deep_ranker import segment_max
    assert torch.allclose(ref, segment_max(h, batch, 5))
    # unsorted / non-contiguous graph ids and an empty graph slot must work; memory stays O(G * max_nodes * d)
    big_h = torch.randn(60_000, 48); big_b = torch.randint(0, 640, (60_000,)); big_b[big_b == 7] = 8
    out = segment_max(big_h, big_b, 640); assert out.shape == (640, 48) and torch.all(out[7] == 0)
    ref_big = torch.full((640, 48), -1e9).scatter_reduce(0, big_b[:, None].expand_as(big_h), big_h, reduce="amax"); ref_big[7] = 0
    assert torch.allclose(ref_big, out)
    assert select_device("auto").type in {"cpu", "cuda"} and select_device("cpu").type == "cpu"


def test_listwise_ce_vectorised_matches_loop():
    torch = pytest.importorskip("torch")
    from mall_space_planner.stage1.rankers.deep_ranker import listwise_softmax_ce
    g = torch.Generator().manual_seed(1); s = torch.randn(50, generator=g); r = torch.rand(50, generator=g) * 4; grp = torch.randint(0, 4, (50,), generator=g); grp[:4] = torch.arange(4)
    ref = sum(-(torch.softmax(r[grp == k], 0) * torch.log_softmax(s[grp == k], 0)).sum() for k in range(4)) / 4
    assert torch.allclose(ref, listwise_softmax_ce(s, r, grp, 4), atol=1e-5)


def test_ar_gnn_set_nll_and_padding():
    torch = pytest.importorskip("torch")
    from mall_space_planner.stage2.generators.ar_gnn import set_nll, _pad
    logits = torch.tensor([0.0, 1.0, 2.0, 0.5, 0.5]); batch = torch.tensor([0, 0, 0, 1, 1]); valid = torch.tensor([False, True, True, True, False])
    p0 = torch.softmax(logits[:3], 0); p1 = torch.softmax(logits[3:], 0)
    ref = (-(p0[1] + p0[2]).log() - p1[0].log()) / 2
    assert torch.allclose(set_nll(logits, batch, 2, valid), ref, atol=1e-6)
    assert _pad(logits, batch, 2).shape == (2, 3)
