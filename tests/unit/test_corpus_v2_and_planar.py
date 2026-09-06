"""Corpus v2 builder (real CSV triplets), planar corridor embedding / decoder, AR-GNN v3 multi-anchor."""

from __future__ import annotations

import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from mall_space_planner.data.corpus_builder import build_corpus, load_any_corpus, load_target_csv
from mall_space_planner.geometry.planar_embed import count_crossings, planar_corridor_embedding
from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyGraph, TopologyPrototype
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.topology.convert import to_networkx

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "graph_csv"


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("gd")
    for p in FIX.glob("*.csv"):
        shutil.copy(p, d / p.name)
    out, stats = build_corpus(d, tmp_path_factory.mktemp("out") / "c.jsonl")
    assert stats.n_written == 1 and stats.n_skeleton_not_subgraph == 0
    return out


def test_target_csv_and_invariants(corpus_path: Path) -> None:
    s = load_any_corpus(corpus_path)[0]
    assert s.sample_id == "B000A0E928_1" and s.mall_id == "B000A0E928" and s.split in {"train", "val", "test"}
    assert s.skeleton.num_nodes == 25 and s.target.num_nodes == 50
    tg = to_networkx(s.target)
    assert set(s.skeleton.nodes) <= set(tg.nodes)
    assert all(tg.has_edge(u, v) for u, v in s.skeleton.edges())  # prototype preserved verbatim
    assert nx.is_connected(tg) and nx.check_planarity(tg)[0]
    assert len(s.skeleton.positions) == 25 and s.target.edge_attrs  # positions + Shared_L_Count carried over
    t = load_target_csv(FIX / "B000A0E928_1_M.csv")
    assert len(t.edges()) == 83


def test_planar_embedding_opens_loops(corpus_path: Path) -> None:
    s = load_any_corpus(corpus_path)[0]
    pos, info = planar_corridor_embedding(s.target)
    assert set(pos) == set(s.target.nodes)
    assert info["crossings"] == 0  # real corridor network is planar and the drawing keeps it so
    assert len(info["faces"]) >= 1  # loop holes -> atrium candidates
    P = np.array(list(pos.values()))
    d = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P))
    assert d.min() > 1e-3  # no coincident nodes (the old "star" collapse)


def test_planar_embedding_tree_is_spine() -> None:
    g = TopologyGraph(adjacency={"A": ["B"], "B": ["C"], "C": ["D"], "D": ["E"], "C": ["F"]})
    pos, info = planar_corridor_embedding(g)
    assert len(info["faces"]) == 0 and count_crossings(to_networkx(g), {k: np.array(v) for k, v in pos.items()}) == 0


def test_planar_corridor_decoder_geometry(corpus_path: Path) -> None:
    from mall_space_planner.evaluation.geometry_eval import GeometryEvaluator
    from mall_space_planner.stage2.decoders import PlanarCorridorDecoder
    from mall_space_planner.stage2.repair.basic import BasicRepairer

    s = load_any_corpus(corpus_path)[0]
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id=s.sample_id, graph=s.target), boundary=SiteBoundary.rectangle(180, 120), constraints=ConstraintSet(target_num_nodes=50, shop_area_min=60, shop_area_max=300), seed=0)
    layout = BasicRepairer().repair(PlanarCorridorDecoder().decode(s.target, req, 0), req)
    res = GeometryEvaluator().evaluate(layout, req.constraints)
    assert res.overall_pass, res.passed
    assert layout.diagnostics["embedding_crossings"] == 0
    assert layout.diagnostics["n_loop_atria"] >= 1 and any(u.kind == "atrium" for u in layout.units)
    assert sum(1 for u in layout.units if u.kind == "shop") >= 25


def test_ar_gnn_v3_multi_anchor_smoke(corpus_path: Path) -> None:
    pytest.importorskip("torch")
    from mall_space_planner.stage2.generators.ar_gnn import ARGNNExpander, canonical_order, teacher_steps

    s = load_any_corpus(corpus_path)[0]
    steps = teacher_steps(s.skeleton, s.target, "label")
    sizes = np.bincount([len(st["valid"]) for st in steps])
    assert sizes[2:].sum() > 0  # multi-anchor steps exist in the real data
    assert len(canonical_order(s.skeleton, s.target, "greedy")) == 25
    gen = ARGNNExpander(d_model=16, n_layers=1, epochs=1, ensemble_k=1, batch_steps=16, device="cpu", val_fraction=0.5, seed=0)
    gen.fit([s] * 4)
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id=s.sample_id, graph=s.skeleton), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=40), seed=0)
    g = gen.generate(req, 0)
    G = to_networkx(g)
    assert G.number_of_nodes() == 40 and nx.is_connected(G) and nx.check_planarity(G)[0]
    assert all(G.has_edge(u, v) for u, v in s.skeleton.edges())
    sk = set(s.skeleton.nodes)
    assert max(G.degree(v) for v in G if v not in sk) <= gen.max_anchors
