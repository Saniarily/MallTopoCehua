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
TOPO_LABEL = {"pred_score": "阶段一预测评分", "num_nodes": "节点数", "num_edges": "连接数", "num_cycles": "回路数", "avg_degree": "平均连接度", "avg_shortest_path": "平均步行路径", "diameter": "拓扑直径",
              "clustering": "聚类系数", "degree_entropy": "连接度熵", "max_betweenness": "最大介数", "n_components": "连通分量", "n_dead_ends": "断头节点", "closeness_mean": "平均接近中心性（整合度）"}
BETTER = {"pred_score": +1, "num_cycles": +1, "avg_shortest_path": -1, "diameter": -1, "max_betweenness": -1, "n_dead_ends": -1, "closeness_mean": +1, "degree_entropy": 0, "clustering": 0, "avg_degree": 0, "n_components": -1, "num_nodes": 0, "num_edges": 0}


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


def make_score_fn(svc, condition):  # noqa: ANN001
    """Stage-1 predicted quality of a *topology* under the mall's own planning conditions: the fitted ranker scores a
    pseudo-candidate whose graph metrics are recomputed from the topology (all other candidate columns = the query mall's
    row). Returns ``None`` when the pipeline has no graph-metric features (score would be constant)."""
    import pandas as pd

    from mall_space_planner.data.adapters import graph_metric_row

    st = svc.stage1
    db = st.ctx.db
    if not db.graph_metric_cols:
        return None
    pool = st.candidate_pool()
    if pool.empty:
        return None
    base = pool.iloc[[0]].copy()
    for c in db.query_cols:
        if c in base:
            base[c] = getattr(condition, c, base[c].iloc[0])
    if "city_cluster" in base and getattr(condition, "city_cluster", None) is not None:
        base["city_cluster"] = condition.city_cluster

    def _score(topology) -> float:  # noqa: ANN001
        row = base.copy()
        for k, v in graph_metric_row(topology).items():
            if k in row:
                row[k] = v
        q_df = st._query_df(condition, 1)
        return float(np.asarray(st.ranker.score(st.ctx, q_df, row.reset_index(drop=True)))[0])

    return _score


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


def renovation_objective(before_ind: dict, cand, score_fn=None) -> float:  # noqa: ANN001
    """Lower is better. A renovation must not lose circulation quality: penalise interior dead ends (beyond the existing
    entrances), fewer loops than the existing network, longer average walking paths, extra components; when a Stage-1
    ``score_fn(topology) -> predicted score`` is given, reward a higher predicted score (weight 4 per score point)."""
    ind = topo_indicators(cand)
    dead = max(0, ind["n_dead_ends"] - before_ind["n_dead_ends"])
    loops = max(0.0, (before_ind["num_cycles"] - ind["num_cycles"]) / max(before_ind["num_cycles"], 1))
    aspl = max(0.0, (ind["avg_shortest_path"] - before_ind["avg_shortest_path"]) / max(before_ind["avg_shortest_path"], 1e-9))
    obj = 1.0 * dead + 3.0 * loops + 3.0 * aspl + 2.0 * max(0, ind["n_components"] - 1)
    if score_fn is not None:
        try:
            obj -= 4.0 * float(score_fn(cand))
        except Exception:  # noqa: BLE001
            pass
    return obj


def existing_entrances(before, gt: dict[str, tuple[float, float]], outline, reach_m: float = 40.0, min_spacing_m: float = 30.0, per_perimeter_m: float = 100.0) -> tuple[dict[str, tuple[float, float]], list[tuple[float, float]]]:  # noqa: ANN001
    """The built floor's entrances. Primary: dead-end key points within ``reach_m`` of the façade. Fallback (the export
    has no dead ends – the entrance sits on the outer loop): the outer-loop nodes closest to the façade, spread along the
    façade (farthest-point), up to ≈ one per ``per_perimeter_m`` of façade (min 2). Returns (node -> real position,
    façade points)."""
    from shapely.geometry import Point

    g = to_networkx(before)
    site = outline.polygon
    nodes: dict[str, tuple[float, float]] = {}
    pts: list[tuple[float, float]] = []

    def _fp(v: str) -> tuple[float, float]:
        q = site.exterior.interpolate(site.exterior.project(Point(gt[v])))
        return (float(q.x), float(q.y))

    for v in g.nodes:
        if g.degree(v) == 1 and v in gt and site.exterior.distance(Point(gt[v])) <= reach_m:
            nodes[v] = gt[v]
            pts.append(_fp(v))
    if not nodes:
        n_target = int(np.clip(round(site.exterior.length / per_perimeter_m), 2, 6))
        cand = sorted((v for v in g.nodes if v in gt and site.exterior.distance(Point(gt[v])) <= reach_m), key=lambda v: site.exterior.distance(Point(gt[v])))
        while cand and len(nodes) < n_target:
            v = cand.pop(0) if not pts else max(cand, key=lambda q: min(np.linalg.norm(np.subtract(_fp(q), p)) for p in pts))
            if v in cand:
                cand.remove(v)
            fp = _fp(v)
            if pts and min(np.linalg.norm(np.subtract(fp, p)) for p in pts) < min_spacing_m:
                break
            nodes[v] = gt[v]
            pts.append(fp)
    return nodes, pts


def select_floors(ds, graph_dir: Path, floors: list[str], cases=None, low_score: float | None = None, min_nodes: int = 12, min_area_m2: float = 6000.0, max_nodes: int = 120, require_entrance: bool = True, prefer_floors: tuple[int, ...] = (1, 2)) -> list[str]:  # noqa: ANN001
    """Renovation candidates: (optionally) low-rated malls, enough key points and floor area, ground / first upper floor
    with at least one dead end near the façade (= an entrance). Floors are ordered ground floor first."""
    from mall_space_planner.data.legacy_adapter import split_floor_id

    low = None
    if cases is not None and low_score is not None and "total_score" in cases:
        low = set(cases.loc[cases["total_score"] <= low_score, "mall_id"].astype(str))
    keep = []
    for fid in floors:
        mall, k = split_floor_id(fid)
        if low is not None and mall not in low:
            continue
        if k is not None and prefer_floors and k not in prefer_floors:
            continue
        m_csv, tot = graph_dir / f"{fid}_M.csv", graph_dir / f"{fid}_total.csv"
        if not (m_csv.exists() and tot.exists() and (graph_dir / f"{fid}_M_simplified.csv").exists()):
            continue
        try:
            before = load_target_csv(m_csv)
            if before.num_nodes < min_nodes or before.num_nodes > max_nodes:
                continue
            outline = ds.outline(fid, 0)
            if outline.area < min_area_m2:
                continue
            if require_entrance:  # a real dead end near the façade (the fallback in existing_entrances is not "evidence")
                gt = m_positions_from_total_csv(tot, outline)
                gb = to_networkx(before)
                from shapely.geometry import Point

                if not any(gb.degree(v) == 1 and v in gt and outline.polygon.exterior.distance(Point(gt[v])) <= 40.0 for v in gb.nodes):
                    continue
        except Exception:  # noqa: BLE001
            continue
        keep.append((0 if k == 1 else 1, fid))
    return [f for _, f in sorted(keep)]


def renovate_floor(fid: str, graph_dir: Path, ds, gen, fitter: CorridorFitter, rp: RenderParams, seed: int = 0, n_candidates: int = 6, score_fn=None, keep_skeleton_positions: bool = False):  # noqa: ANN001, ANN201
    """Renovation of one built floor.

    * kept: outline, **entrance positions** (dead ends near the façade, anchored) and – as topology, not as
      geometry – the Stage-1 skeleton (the generator keeps skeleton nodes / edges; their *positions* are re-optimised so the
      new network can take a better shape). ``keep_skeleton_positions=True`` restores the earlier behaviour (all skeleton
      nodes anchored).
    * regrown: the rest of the network at the existing size and density; ``n_candidates`` samples ranked by
      :func:`renovation_objective` (+ the Stage-1 predicted score when ``score_fn`` is given).
    """
    from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyMetrics, TopologyPrototype
    from mall_space_planner.stage2.base import GenerationRequest
    from mall_space_planner.stage3.evaluate import evaluate_fit

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
    real_cr = outline.extra.get("real_corridor_ratio") if hasattr(outline, "extra") else None
    if real_cr and np.isfinite(real_cr):
        rp = RenderParams(**{**rp.__dict__, "corridor_ratio": float(np.clip(real_cr, rp.corridor_ratio, 0.28))})
    # ---- regrow at the existing size and density
    req = GenerationRequest(prototype=TopologyPrototype(prototype_id=fid, graph=skeleton), boundary=SiteBoundary.rectangle(100, 100),
                            constraints=ConstraintSet(target_num_nodes=before.num_nodes, target_metrics=TopologyMetrics(avg_degree=2.0 * before.num_edges / max(before.num_nodes, 1))), seed=seed)
    ind_b0 = topo_indicators(before)
    cands = [gen.generate(req, seed + k) for k in range(max(n_candidates, 1))]
    objs = [renovation_objective(ind_b0, c, score_fn) for c in cands]
    after = cands[int(np.argmin(objs))]
    # ---- anchors: existing entrances (always); skeleton positions only on request
    ent_nodes, ent_pts = existing_entrances(before, gt, outline)
    anchors = {v: gt[v] for v in ent_nodes if v in set(after.nodes)}
    if keep_skeleton_positions:
        anchors.update({v: gt[v] for v in sk_nodes if v in gt})
    res = fitter.fit(after, outline, seed=seed, skeleton_nodes=sk_nodes, anchors=anchors, entrance_targets=ent_pts, init="anchored" if keep_skeleton_positions else "outline")
    plan = render_corridors(after, res.positions, outline, res.roles, rp, skeleton_nodes=sk_nodes, entrance_points=ent_pts)
    ev = evaluate_fit(after, res, plan, outline)
    # ---- before: real positions through the same renderer (like-for-like corridor comparison)
    gt_b = {v: gt[v] for v in before.nodes if v in gt}
    plan_b = None
    if len(gt_b) == before.num_nodes:
        try:
            plan_b = render_corridors(before, gt_b, outline, None, rp, skeleton_nodes=sk_nodes)
        except Exception:  # noqa: BLE001
            plan_b = None
    ind_b, ind_a = topo_indicators(before), topo_indicators(after)
    if score_fn is not None:
        try:
            ind_b["pred_score"], ind_a["pred_score"] = float(score_fn(before)), float(score_fn(after))
        except Exception:  # noqa: BLE001
            pass
    g_b = to_networkx(before)
    row = {"floor_id": fid, "n_nodes": before.num_nodes, "n_skeleton": len(sk_nodes), "n_anchored": res.diagnostics.get("n_anchored"), "n_existing_entrances": len(ent_pts), "outline_area_m2": outline.area,
           **{f"before_{k}": v for k, v in ind_b.items()}, **{f"after_{k}": v for k, v in ind_a.items()},
           "before_sharp_angle_rate": sharp_angle_rate(g_b, {k: np.asarray(v) for k, v in gt_b.items()}, fitter.p.min_angle_deg) if len(gt_b) == before.num_nodes else None,
           "after_sharp_angle_rate": ev.get("sharp_angle_rate"), "after_crossings": ev.get("crossings"), "after_inside_ratio": ev.get("inside_ratio"),
           "after_corridor_ratio": plan.diagnostics["corridor_ratio"], "after_n_entrances": plan.diagnostics["n_entrances"], "after_n_atria": plan.diagnostics["n_atria"], "after_n_vertical_cores": plan.diagnostics["n_vertical_cores"],
           "before_corridor_ratio": plan_b.diagnostics["corridor_ratio"] if plan_b else None, "before_n_entrances": plan_b.diagnostics["n_entrances"] if plan_b else None, "before_n_atria": plan_b.diagnostics["n_atria"] if plan_b else None,
           "n_candidates": len(cands), "objective_best": float(min(objs)), "objective_worst": float(max(objs))}
    return dict(before=before, after=after, skeleton=skeleton, sk_nodes=sk_nodes, anchored=set(anchors), entrance_points=ent_pts, outline=outline, gt=gt_b, res=res, plan=plan, plan_b=plan_b, row=row, ind_b=ind_b, ind_a=ind_a)


__all__ = ["BETTER", "TOPO_KEYS", "TOPO_LABEL", "build_generator", "existing_entrances", "make_score_fn", "renovate_floor", "renovation_objective", "select_floors", "topo_indicators"]
