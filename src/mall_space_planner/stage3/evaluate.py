"""Stage-3 metrics: geometric sanity of the fitted corridor network + agreement with the real drawing.

Geometric (always available):
  inside_ratio, crossings, ortho_deviation_deg (mean edge angle to the wall frame), spacing_violation_rate,
  served_area_ratio (share of the inset within one shop depth of a corridor), corridor_ratio (corridor area /
  floor area; real malls ≈ 0.12–0.25), n_entrances, mean_edge_length_m, min_edge_length_m.

Against ground truth (same floor, real M ``CenterPoint``):
  procrustes_rmse_m  – RMSE after optimal similarity alignment (scale+rotation+translation): shape agreement
                       *with node correspondence* (strict: the same M id must land at the same place);
  rigid_rmse_m       – RMSE after rotation+translation only (also penalises wrong overall scale);
  chamfer_m          – symmetric nearest-neighbour distance between fitted and real key points in the outline
                       frame (label-free: "are the corridors where the real corridors are");
  gt_outer_facade_dist_m – mean façade distance of the real outer-loop nodes (compare with outer_facade_dist_m).

Note: on elongated floors Procrustes is dominated by the long axis, so even random placements score
≈ 0.2 × diagonal; use ``chamfer_m`` and ``*_rand`` baselines (``random_baseline``) for a fair comparison.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from mall_space_planner.geometry.planar_embed import count_crossings
from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.stage3.fit import FitResult
from mall_space_planner.stage3.outline import Outline
from mall_space_planner.stage3.render import CorridorPlan
from mall_space_planner.topology.convert import to_networkx


def procrustes(X: np.ndarray, Y: np.ndarray, allow_scale: bool = True) -> tuple[float, dict]:
    """Align X onto Y (rows correspond). Returns (rmse, params)."""
    mx, my = X.mean(0), Y.mean(0)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = np.linalg.svd(X0.T @ Y0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max((X0**2).sum(), 1e-12) if allow_scale else 1.0
    Xa = s * X0 @ R + my
    rmse = float(np.sqrt(((Xa - Y) ** 2).sum(1).mean()))
    return rmse, {"scale": float(s), "rotation_deg": float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))}


def chamfer(X: np.ndarray, Y: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance (label-free)."""
    from scipy.spatial import cKDTree

    a = cKDTree(Y).query(X)[0]
    b = cKDTree(X).query(Y)[0]
    return float(0.5 * (a.mean() + b.mean()))


def random_baseline(outline: Outline, Y: np.ndarray, n_trials: int = 20, seed: int = 0) -> dict[str, float]:
    """Uniform random placement inside the outline bbox: reference values for procrustes / chamfer."""
    rng = np.random.RandomState(seed)
    minx, miny, maxx, maxy = outline.polygon.bounds
    pr, ch = [], []
    for _ in range(n_trials):
        X = np.c_[rng.uniform(minx, maxx, len(Y)), rng.uniform(miny, maxy, len(Y))]
        pr.append(procrustes(X, Y)[0])
        ch.append(chamfer(X, Y))
    return {"procrustes_rmse_rand_m": float(np.mean(pr)), "chamfer_rand_m": float(np.mean(ch))}


def evaluate_fit(topology: TopologyGraph, res: FitResult, plan: CorridorPlan, outline: Outline, gt_positions: dict[str, tuple[float, float]] | None = None) -> dict:
    g = to_networkx(topology)
    pos = {k: np.asarray(v) for k, v in res.positions.items()}
    site = outline.polygon
    out = dict(res.diagnostics)
    out["crossings"] = count_crossings(g, pos)
    out["inside_ratio"] = float(np.mean([site.contains(Point(p)) or site.touches(Point(p)) for p in pos.values()]))
    lens = [float(np.linalg.norm(pos[u] - pos[v])) for u, v in g.edges]
    out["mean_edge_length_m"] = float(np.mean(lens)) if lens else 0.0
    out["min_edge_length_m"] = float(np.min(lens)) if lens else 0.0
    out.update({k: v for k, v in plan.diagnostics.items()})
    out["outer_facade_dist_m"] = float(np.mean([site.exterior.distance(Point(pos[v])) for v in g.nodes if res.roles.get(v) == "outer"])) if any(r == "outer" for r in res.roles.values()) else float("nan")
    if gt_positions:
        keys = [k for k in g.nodes if k in gt_positions]
        if len(keys) >= 3:
            X = np.array([pos[k] for k in keys])
            Y = np.array([gt_positions[k] for k in keys])
            out["procrustes_rmse_m"], prm = procrustes(X, Y, True)
            out["rigid_rmse_m"], _ = procrustes(X, Y, False)
            out["procrustes_scale"] = prm["scale"]
            out["chamfer_m"] = chamfer(X, Y)
            out.update(random_baseline(outline, Y))
            out["gt_outer_facade_dist_m"] = float(np.mean([site.exterior.distance(Point(gt_positions[v])) for v in keys if res.roles.get(v) == "outer"])) if any(res.roles.get(v) == "outer" for v in keys) else float("nan")
            out["gt_crossings"] = count_crossings(g.subgraph(keys), {k: np.asarray(gt_positions[k]) for k in keys})
            # normalised: RMSE relative to the outline diagonal
            minx, miny, maxx, maxy = site.bounds
            out["procrustes_rmse_norm"] = out["procrustes_rmse_m"] / float(np.hypot(maxx - minx, maxy - miny))
    out.pop("roles", None)
    return out


__all__ = ["chamfer", "evaluate_fit", "procrustes", "random_baseline"]
