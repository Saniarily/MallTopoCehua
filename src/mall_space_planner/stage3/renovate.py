"""Stage 3 – renovation mode: same outline, existing main corridors and entrances kept (anchored at their real
positions), the secondary network regrown by the Stage-2 generator and rendered as a corridor plan; before / after
topology indicators for the comparison table. Used by ``scripts/renovate_stage3.py`` and figure F09."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mall_space_planner.data.corpus_builder import load_target_csv
from mall_space_planner.data.legacy_adapter import load_graph_csv
from mall_space_planner.stage3.fit import CorridorFitter, sharp_angle_rate
from mall_space_planner.stage3.outline import m_positions_from_total_csv
from mall_space_planner.stage3.render import RenderParams, render_corridors
from mall_space_planner.topology.convert import to_networkx
from mall_space_planner.topology.metrics import compute_topology_metrics

TOPO_KEYS = ["num_nodes", "num_edges", "num_cycles", "avg_degree", "avg_shortest_path", "diameter", "clustering", "degree_entropy", "max_betweenness", "n_components"]
TOPO_LABEL = {"num_nodes": "节点数", "num_edges": "连接数", "num_cycles": "回路数", "avg_degree": "平均连接度", "avg_shortest_path": "平均步行路径", "diameter": "拓扑直径",
              "clustering": "聚类系数", "degree_entropy": "连接度熵", "max_betweenness": "最大介数", "n_components": "连通分量", "n_dead_ends": "断头节点", "closeness_mean": "平均接近中心性（整合度）"}
BETTER = {"num_cycles": +1, "avg_shortest_path": -1, "diameter": -1, "max_betweenness": -1, "n_dead_ends": -1, "closeness_mean": +1, "degree_entropy": 0, "clustering": 0, "avg_degree": 0, "n_components": -1, "num_nodes": 0, "num_edges": 0}


def topo_indicators(topo) -> dict:  # noqa: ANN001
    g = to_networkx(topo)
    m = compute_topology_metrics(topo).model_dump()
    out = {k: m.get(k) for k in TOPO_KEYS}
    out["n_dead_ends"] = int(sum(1 for v in g.nodes if g.degree(v) == 1))
    import networkx as nx

    sub = g.subgraph(max(nx.connected_components(g), key=len)) if g.number_of_nodes() else g
    cc = nx.closeness_centrality(sub) if sub.number_of_nodes() > 1 else {}
    out["closeness_mean"] = float(np.mean(list(cc.values()))) if cc else 0.0
    return out


def build_generator(results: Path | None):  # noqa: ANN201
    from mall_space_planner.registry import build
    import mall_space_planner.stage2.generators  # noqa: F401

    for ck in [Path("outputs/checkpoints/stage2/stage2_ar_gnn"), (results or Path("data/results_snapshot")) / "stage2/checkpoints/ar_gnn"]:
        if (ck / "ar_gnn.pt").exists():
            try:
                return build("generator", {"name": "ar_gnn", "params": {"checkpoint": str(ck), "best_of": 16, "temperature": 0.7, "device": "cpu"}}), "AR-GNN + 16 次择优"
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] ar_gnn checkpoint unusable ({exc}); falling back to rule best-of-16")
    return build("generator", {"name": "search_expander", "params": {"n_trials": 16, "w_aspl": 1.5}}), "规则 + 16 次择优"


def renovation_objective(before_ind: dict, cand) -> float:  # noqa: ANN001
    """Lower is better. A renovation must not lose circulation quality: penalise interior dead ends (the built floor has
    none inside), fewer loops than the existing network, longer average walking paths and extra components."""
    ind = topo_indicators(cand)
    dead = max(0, ind["n_dead_ends"] - before_ind["n_dead_ends"])
    loops = max(0.0, (before_ind["num_cycles"] - ind["num_cycles"]) / max(before_ind["num_cycles"], 1))
    aspl = max(0.0, (ind["avg_shortest_path"] - before_ind["avg_shortest_path"]) / max(before_ind["avg_shortest_path"], 1e-9))
    return 1.0 * dead + 3.0 * loops + 3.0 * aspl + 2.0 * max(0, ind["n_components"] - 1)


def renovate_floor(fid: str, graph_dir: Path, ds, gen, fitter: CorridorFitter, rp: RenderParams, seed: int = 0, n_candidates: int = 6):  # noqa: ANN001, ANN201
    from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype
    from mall_space_planner.stage2.base import GenerationRequest

    m_csv, sk_csv, sk_attr, tot = graph_dir / f"{fid}_M.csv", graph_dir / f"{fid}_M_simplified.csv", graph_dir / f"{fid}_M_simplified_node_attributes.csv", graph_dir / f"{fid}_total.csv"
    for pth in (m_csv, sk_csv, sk_attr, tot):
        if not pth.exists():
            raise FileNotFoundError(pth)
    before = load_target_csv(m_csv)
    skeleton = load_graph_csv(sk_csv, sk_attr)
    outline = ds.outline(fid, 0) if ds.paths.graph_dir else None
    if outline is None:
        from mall_space_planner.stage3.outline import outline_from_total_csv

        outline = outline_from_total_csv(tot, close_px=20)
    gt = m_positions_from_total_csv(tot, outline)
    sk_nodes = set(skeleton.nodes) & set(before.nodes)
    # renovation keeps the existing corridor-area budget: use the floor's real corridor ratio when known (clipped to
    # the real distribution's IQR upper end so a corridor-heavy plan does not get 8 m secondary corridors)
    real_cr = outline.extra.get("real_corridor_ratio") if hasattr(outline, "extra") else None
    if real_cr and np.isfinite(real_cr):
        rp = RenderParams(**{**rp.__dict__, "corridor_ratio": float(np.clip(real_cr, rp.corridor_ratio, 0.28))})
    # ---- regrow
    from mall_space_planner.schemas import TopologyMetrics

    # same size and the *existing* network's density as target (a renovation keeps the corridor budget)
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id=fid, graph=skeleton), boundary=SiteBoundary.rectangle(100, 100),
                            constraints=ConstraintSet(target_num_nodes=before.num_nodes, target_metrics=TopologyMetrics(avg_degree=2.0 * before.num_edges / max(before.num_nodes, 1))), seed=seed)
    ind_b0 = topo_indicators(before)
    # several generator samples; keep the one that best preserves circulation quality (see renovation_objective)
    cands = [gen.generate(req, seed + k) for k in range(max(n_candidates, 1))]
    after = min(cands, key=lambda c: renovation_objective(ind_b0, c))
    # ---- fit with anchored skeleton (real positions), same outline
    anchors = {v: gt[v] for v in sk_nodes if v in gt}
    res = fitter.fit(after, outline, seed=seed, skeleton_nodes=sk_nodes, anchors=anchors)
    plan = render_corridors(after, res.positions, outline, res.roles, rp, skeleton_nodes=sk_nodes)
    from mall_space_planner.stage3.evaluate import evaluate_fit

    ev = evaluate_fit(after, res, plan, outline)
    # ---- before: real positions, same renderer (so the corridor-plan comparison is like for like)
    gt_b = {v: gt[v] for v in before.nodes if v in gt}
    plan_b = None
    if len(gt_b) == before.num_nodes:
        try:
            plan_b = render_corridors(before, gt_b, outline, None, rp, skeleton_nodes=sk_nodes)
        except Exception:  # noqa: BLE001
            plan_b = None
    ind_b, ind_a = topo_indicators(before), topo_indicators(after)
    g_b = to_networkx(before)
    row = {"floor_id": fid, "n_nodes": before.num_nodes, "n_skeleton": len(sk_nodes), "n_anchored": res.diagnostics.get("n_anchored"), "outline_area_m2": outline.area,
           **{f"before_{k}": v for k, v in ind_b.items()}, **{f"after_{k}": v for k, v in ind_a.items()},
           "before_sharp_angle_rate": sharp_angle_rate(g_b, {k: np.asarray(v) for k, v in gt_b.items()}, fitter.p.min_angle_deg) if len(gt_b) == before.num_nodes else None,
           "after_sharp_angle_rate": ev.get("sharp_angle_rate"), "after_crossings": ev.get("crossings"), "after_inside_ratio": ev.get("inside_ratio"),
           "after_corridor_ratio": plan.diagnostics["corridor_ratio"], "after_n_entrances": plan.diagnostics["n_entrances"], "after_n_atria": plan.diagnostics["n_atria"], "after_n_vertical_cores": plan.diagnostics["n_vertical_cores"],
           "before_corridor_ratio": plan_b.diagnostics["corridor_ratio"] if plan_b else None, "before_n_entrances": plan_b.diagnostics["n_entrances"] if plan_b else None, "before_n_atria": plan_b.diagnostics["n_atria"] if plan_b else None}
    return dict(before=before, after=after, skeleton=skeleton, sk_nodes=sk_nodes, outline=outline, gt=gt_b, res=res, plan=plan, plan_b=plan_b, row=row, ind_b=ind_b, ind_a=ind_a)



__all__ = ["BETTER", "TOPO_KEYS", "TOPO_LABEL", "renovate_floor", "renovation_objective", "topo_indicators"]
