"""Planning workbench: the three-stage workflow behind UI 3, UI-independent.

Two entry paths share the same Stage-2 generator and Stage-3 fitter / renderer:

* **new build**: conditions → type ranking → Top-K prototypes → expand the chosen skeleton to a complete key-point
  network (several candidates) → fit into an uploaded / drawn / database outline → corridor plan;
* **renovation**: pick a built floor → keep outline + entrances → regrow → corridor plan → before / after indicators.

Every candidate carries its own seed, so a single candidate can be regenerated ("one-click re-roll") without touching
the others. Results are plain dataclasses (shapely geometries + TopologyGraph); exporters turn them into
JSON / GeoJSON / SVG / PNG.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mall_space_planner.api.service import PlanningService
from mall_space_planner.hub.catalog import Catalog
from mall_space_planner.schemas import ConstraintSet, PlanningCondition, Recommendation, SiteBoundary, TopologyGraph, TopologyMetrics, TopologyPrototype
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.stage3 import CorridorFitter, FitParams, Outline, RenderParams, outline_from_points, render_corridors
from mall_space_planner.stage3.evaluate import evaluate_fit
from mall_space_planner.stage3.renovate import BETTER, TOPO_LABEL, build_generator, make_score_fn, renovate_floor, select_floors, topo_indicators
from mall_space_planner.topology.convert import to_networkx
from mall_space_planner.utils.config import resolve_config

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Candidate:
    index: int
    seed: int
    topology: TopologyGraph
    fit: Any  # FitResult
    plan: Any  # CorridorPlan
    metrics: dict[str, Any]
    indicators: dict[str, Any]
    pred_score: float | None = None
    outline: Outline | None = None
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        d = self.plan.diagnostics
        return {
            "候选": self.index + 1, "seed": self.seed, "节点": self.topology.num_nodes, "连接": self.topology.num_edges,
            "回路": self.indicators.get("num_cycles"), "平均步行路径": round(self.indicators.get("avg_shortest_path") or 0, 2),
            "整合度": round(self.indicators.get("closeness_mean") or 0, 3), "断头": self.indicators.get("n_dead_ends"),
            "交叉": self.metrics.get("crossings"), "锐角率": round(self.metrics.get("sharp_angle_rate") or 0, 2),
            "走廊占比": round(d.get("corridor_ratio", 0), 2), "出入口": d.get("n_entrances"), "竖向核": d.get("n_vertical_cores"), "中庭": d.get("n_atria"),
            "预测评分": None if self.pred_score is None else round(self.pred_score, 3), "耗时 s": round(self.elapsed_s, 1),
        }


@dataclass
class Workbench:
    catalog: Catalog
    svc: PlanningService
    generator: Any
    generator_name: str
    stage3_cfg: dict[str, Any] = field(default_factory=dict)

    # ---- construction --------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, catalog: Catalog | None = None, stage1_config: str = "configs/stage1/extra_trees.yaml", stage2_config: str = "configs/stage2/search_baseline.yaml") -> Workbench:
        cat = catalog or Catalog.load()
        s1 = resolve_config(stage1_config, [])
        s2 = resolve_config(stage2_config, [])
        s1["stage1"]["counterfactuals"] = {"enabled": True, "deltas": {"total_area": 0.3, "count_1km": 0.5}}
        svc = PlanningService(cat.db, s1, s2)
        gen, name = build_generator(ROOT / "data/results_snapshot")
        try:
            s3 = resolve_config("configs/data/legacy.yaml", []).get("stage3") or {}
        except Exception:  # noqa: BLE001
            s3 = {}
        return cls(catalog=cat, svc=svc, generator=gen, generator_name=name, stage3_cfg=s3)

    # ---- stage 1 ---------------------------------------------------------------------------------------------------
    def default_condition(self, cluster: int = 2) -> PlanningCondition:
        df = self.catalog.cases
        sub = df[df["city_cluster"] == cluster] if "city_cluster" in df and (df["city_cluster"] == cluster).any() else df
        med = sub[self.catalog.db.query_cols].median()
        return PlanningCondition(city_cluster=cluster, **{c: float(med[c]) for c in self.catalog.db.query_cols})

    def condition_from_case(self, floor_id: str) -> PlanningCondition:
        r = self.catalog.case(floor_id)
        return PlanningCondition(city_cluster=int(r["city_cluster"]) if r.get("city_cluster") is not None else None, **{c: (float(r[c]) if r.get(c) is not None else None) for c in self.catalog.db.query_cols})

    def recommend_types(self, cond: PlanningCondition):  # noqa: ANN201
        return self.svc.recommend_types(cond)

    def recommend(self, cond: PlanningCondition, layout_type: str | None, top_k: int = 5) -> list[Recommendation]:
        if layout_type:
            return self.svc.recommend_within_type(cond, layout_type, top_k=top_k)
        return self.svc.recommend(cond, top_k=top_k, with_counterfactuals=True)

    def score_fn(self, cond: PlanningCondition):  # noqa: ANN201
        return make_score_fn(self.svc, cond)

    # ---- stage 2 + 3 (new build) --------------------------------------------------------------------------------------
    def outline_from(self, points: list[tuple[float, float]] | None = None, floor_id: str | None = None, similar_area: float | None = None) -> tuple[Outline, str]:
        """Outline from hand-drawn points, a database floor, or the closest-area real floor."""
        if points:
            return outline_from_points(points), "hand-drawn"
        if floor_id:
            o = self.catalog.outline(floor_id)
            if o is not None:
                return o, f"floor {floor_id}"
        if similar_area:
            try:
                from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths

                ds = Stage3Dataset(Stage3Paths.from_config(resolve_config("configs/data/legacy.yaml", [])))
                c = ds.similar_outlines(similar_area, k=1)
                if c:
                    return ds.outline(c[0], 0), f"similar floor {c[0]}"
            except Exception:  # noqa: BLE001
                pass
        # fallback: rectangle with mall-like aspect
        A = similar_area or 15000.0
        w = float(np.sqrt(A * 2.2)); h = A / w
        return outline_from_points([(0, 0), (w, 0), (w, h), (0, h)]), "rectangle (fallback)"

    def _fitter(self, restarts: int = 4, iters: int = 100) -> CorridorFitter:
        return CorridorFitter(FitParams(shop_depth=float(self.stage3_cfg.get("shop_depth_m", 14.0)), n_restarts=restarts, iters=iters))

    def _render_params(self, corridor_ratio: float | None = None, min_entrances: int = 2, max_entrances: int = 6, max_atria: int = 4) -> RenderParams:
        return RenderParams(corridor_ratio=float(corridor_ratio or self.stage3_cfg.get("corridor_ratio", 0.18)), min_entrances=min_entrances, max_entrances=max_entrances, max_atria=max_atria)

    def generate_candidate(self, skeleton: TopologyGraph, prototype_id: str, outline: Outline, n_target: int, seed: int, index: int = 0,
                           layout_type: str | None = None, target_avg_degree: float | None = None, corridor_ratio: float | None = None,
                           min_entrances: int = 2, max_entrances: int = 6, max_atria: int = 4, score_fn=None, entrance_points: list[tuple[float, float]] | None = None,
                           anchors: dict[str, tuple[float, float]] | None = None, restarts: int = 4, iters: int = 100) -> Candidate:  # noqa: ANN001
        t0 = time.time()
        tm = TopologyMetrics(avg_degree=target_avg_degree) if target_avg_degree else None
        req = GenerationRequest(prototype=TopologyPrototype(prototype_id=prototype_id, graph=skeleton, layout_type=layout_type), boundary=SiteBoundary.rectangle(100, 100),
                                constraints=ConstraintSet(target_num_nodes=max(n_target, skeleton.num_nodes), target_metrics=tm, min_entrances=min_entrances, num_atria=max_atria), seed=seed)
        topo = self.generator.generate(req, seed)
        sk_nodes = set(skeleton.nodes)
        res = self._fitter(restarts, iters).fit(topo, outline, seed=seed, skeleton_nodes=sk_nodes, anchors=anchors, entrance_targets=entrance_points, init="outline")
        plan = render_corridors(topo, res.positions, outline, res.roles, self._render_params(corridor_ratio, min_entrances, max_entrances, max_atria), skeleton_nodes=sk_nodes, entrance_points=entrance_points)
        ev = evaluate_fit(topo, res, plan, outline)
        ps = None
        if score_fn is not None:
            try:
                ps = float(score_fn(topo))
            except Exception:  # noqa: BLE001
                ps = None
        return Candidate(index=index, seed=seed, topology=topo, fit=res, plan=plan, metrics=ev, indicators=topo_indicators(topo), pred_score=ps, outline=outline, elapsed_s=time.time() - t0)

    def generate_candidates(self, skeleton: TopologyGraph, prototype_id: str, outline: Outline, n_target: int, n_candidates: int = 3, seed: int = 0, **kw: Any) -> list[Candidate]:
        return [self.generate_candidate(skeleton, prototype_id, outline, n_target, seed + i, index=i, **kw) for i in range(n_candidates)]

    # ---- renovation ----------------------------------------------------------------------------------------------------
    def _stage3_dataset(self):  # noqa: ANN202
        from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths

        ds = Stage3Dataset(Stage3Paths.from_config(resolve_config("configs/data/legacy.yaml", [])))
        gd = ds.paths.graph_dir
        if gd is None or not gd.exists() or not any(gd.glob("*_M.csv")):
            gd = ROOT / "tests/fixtures/graph_csv"
        ds.paths.graph_dir = gd
        return ds, gd

    def renovation_floor_ids(self) -> list[str]:
        """Every built floor whose graph CSVs (M / M_simplified / total) are reachable – no filtering."""
        try:
            _, gd = self._stage3_dataset()
            return sorted({p.name[: -len("_M.csv")] for p in gd.glob("*_M.csv") if not p.name.endswith("_M_simplified.csv") and (gd / f"{p.name[: -len('_M.csv')]}_total.csv").exists()})
        except Exception:  # noqa: BLE001
            return []

    def renovation_floors(self, low_score: float | None = None, limit: int = 100, min_nodes: int = 12, min_area: float = 6000.0, prefer_floors: tuple[int, ...] | None = (1, 2), require_entrance: bool = True, splits: tuple[str, ...] | None = None) -> list[str]:
        """Floors recommended for the renovation workflow (filter on nodes / area / floor index / entrance / score).
        ``splits=None`` = all floors (default since the designer picks the case); pass ("test",) to restrict."""
        try:
            ds, gd = self._stage3_dataset()
            ids = self.renovation_floor_ids()
            df = self.catalog.cases
            if splits and "split" in df:
                keep = set(df.loc[df["split"].isin(splits), self.catalog.db.id_col].astype(str))
                ids = [i for i in ids if i in keep] or ids
            return select_floors(ds, gd, ids, cases=df, low_score=low_score, min_nodes=min_nodes, min_area_m2=min_area, require_entrance=require_entrance, prefer_floors=tuple(prefer_floors) if prefer_floors else ())[:limit]
        except Exception:  # noqa: BLE001
            return []

    def renovation_floor_table(self, floor_ids: list[str]) -> list[dict[str, Any]]:
        """Per-floor metadata for the floor picker: mall score / type / area / node count / floor index / entrances.
        Uses ``outputs/floor_plates/floors.csv`` (scripts/export_floor_plates.py) when present – instant; otherwise the
        graph CSVs are parsed (≈ 0.1 s per floor, cached in memory for the session)."""
        from mall_space_planner.data.legacy_adapter import split_floor_id
        from mall_space_planner.hub.plates import load_plates_table

        df = self.catalog.cases
        pre = load_plates_table()
        pre_rows = {str(r["floor_id"]): r for _, r in pre.iterrows()} if pre is not None else {}
        cache = self.__dict__.setdefault("_floor_meta_cache", {})
        rows = []
        for fid in floor_ids:
            mall, k = split_floor_id(fid)
            row = {"floor_id": fid, "mall_id": mall, "floor": k}
            case = df[df[self.catalog.db.id_col].astype(str) == fid]
            if not case.empty:
                c = case.iloc[0]
                row.update({"score": c.get(self.catalog.db.label_col), "layout_type": c.get("layout_type"), "split": c.get("split")})
            if fid in pre_rows:
                r = pre_rows[fid]
                row.update({kk: r.get(kk) for kk in ("n_nodes", "area_m2", "n_entrances", "nodes_outside", "align_mode") if kk in r})
            elif fid in cache:
                row.update(cache[fid])
            else:
                try:
                    from shapely.geometry import Point

                    from mall_space_planner.hub.plates import load_floor

                    ds, gd = self._stage3_dataset()
                    rn = load_floor(fid, ds, gd)
                    g = to_networkx(rn["full"]); gt = rn["positions"]; outline = rn["outline"]
                    meta = {"n_nodes": rn["full"].num_nodes, "area_m2": round(outline.area), "n_entrances": int(sum(1 for v in g.nodes if g.degree(v) == 1 and v in gt and outline.polygon.exterior.distance(Point(gt[v])) <= 40.0))}
                    cache[fid] = meta; row.update(meta)
                except Exception:  # noqa: BLE001
                    pass
            rows.append(row)
        return rows

    def floor_thumbnail(self, floor_id: str, size_px: int = 360, use_cache: bool = True) -> bytes | None:
        """Square PNG for the floor picker. Order: pre-rendered ``outputs/floor_plates/thumbs/<floor>.png`` (from
        ``scripts/export_floor_plates.py``) → disk cache ``outputs/cache/thumbnails`` → render now (colour-block plan as
        background when reachable, full M network on top, outline frame) and cache."""
        from mall_space_planner.hub.plates import find_prerendered_thumbnail, load_floor, render_thumbnail_bytes

        pre = find_prerendered_thumbnail(floor_id)
        if pre is not None:
            return pre.read_bytes()
        cache = ROOT / "outputs" / "cache" / "thumbnails" / f"{floor_id}_{size_px}.png"
        if use_cache and cache.exists():
            return cache.read_bytes()
        try:
            ds, gd = self._stage3_dataset()
            rn = load_floor(floor_id, ds, gd)
        except Exception:  # noqa: BLE001
            return None
        data = render_thumbnail_bytes(rn, size_px)
        if use_cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True); cache.write_bytes(data)
            except OSError:
                pass
        return data

    def renovate(self, floor_id: str, seed: int = 0, n_candidates: int = 6, use_score: bool = True, keep_skeleton_positions: bool = False, restarts: int = 4, iters: int = 100) -> dict[str, Any]:
        ds, gd = self._stage3_dataset()
        if not (gd / f"{floor_id}_M.csv").exists():
            raise FileNotFoundError(f"{floor_id}_M.csv not found under {gd}")
        score_fn = None
        if use_score:
            try:
                score_fn = self.score_fn(self.condition_from_case(floor_id))
            except Exception:  # noqa: BLE001
                score_fn = None
        r = renovate_floor(floor_id, gd, ds, self.generator, self._fitter(restarts, iters), self._render_params(), seed=seed, n_candidates=n_candidates, score_fn=score_fn, keep_skeleton_positions=keep_skeleton_positions)
        r["generator_name"] = self.generator_name
        r["table"] = self.indicator_table(r["ind_b"], r["ind_a"], r["row"])
        return r

    @staticmethod
    def indicator_table(ind_b: dict, ind_a: dict, row: dict | None = None) -> list[dict[str, Any]]:
        rows = []
        for k in ["pred_score", "num_cycles", "avg_shortest_path", "diameter", "closeness_mean", "max_betweenness", "degree_entropy", "avg_degree", "n_dead_ends"]:
            b, a = ind_b.get(k), ind_a.get(k)
            if b is None or a is None:
                continue
            better = BETTER.get(k, 0)
            arrow = "" if better == 0 or abs(a - b) < 1e-9 else ("▲" if (a - b) * better > 0 else "▼")
            rows.append({"指标": TOPO_LABEL.get(k, k), "现状": round(b, 3) if isinstance(b, float) else b, "更新": round(a, 3) if isinstance(a, float) else a, "": arrow})
        if row and row.get("before_sharp_angle_rate") is not None and row.get("after_sharp_angle_rate") is not None:
            b, a = row["before_sharp_angle_rate"], row["after_sharp_angle_rate"]
            rows.append({"指标": "锐角(<60°)比例", "现状": round(b, 3), "更新": round(a, 3), "": "" if abs(a - b) < 1e-9 else ("▲" if a < b else "▼")})
        return rows

    # ---- constraint check & risks ---------------------------------------------------------------------------------------
    @staticmethod
    def constraint_report(c: Candidate, min_entrances: int = 2, max_sharp: float = 0.2, corridor_ratio_range: tuple[float, float] = (0.12, 0.28)) -> list[dict[str, Any]]:
        d = c.plan.diagnostics
        checks = [
            ("平面（无交叉）", c.metrics.get("crossings", 0) == 0, f"交叉 {c.metrics.get('crossings')}"),
            ("全部在轮廓内（节点 + 走廊）", (c.metrics.get("inside_ratio") or 0) >= 0.999 and not c.metrics.get("edges_outside"), f"节点 {(c.metrics.get('inside_ratio') or 0) * 100:.0f}% · 越界走廊 {c.metrics.get('edges_outside', 0)}"),
            ("出入口 ≥ 最小值", d.get("n_entrances", 0) >= min_entrances, f"{d.get('n_entrances')} / {min_entrances}"),
            ("无内部断头（除竖向核）", c.indicators.get("n_components", 1) == 1, f"连通分量 {c.indicators.get('n_components')}"),
            ("锐角率 ≤ 阈值", (c.metrics.get("sharp_angle_rate") or 0) <= max_sharp, f"{c.metrics.get('sharp_angle_rate', 0):.2f} / {max_sharp}"),
            ("走廊占比在真实区间", corridor_ratio_range[0] <= d.get("corridor_ratio", 0) <= corridor_ratio_range[1], f"{d.get('corridor_ratio', 0):.2f} ∈ [{corridor_ratio_range[0]}, {corridor_ratio_range[1]}]"),
            ("有中庭", d.get("n_atria", 0) >= 1, f"{d.get('n_atria')}"),
        ]
        return [{"约束": n, "满足": ok, "值": v} for n, ok, v in checks]

    @staticmethod
    def risks(c: Candidate) -> list[str]:
        out = []
        d = c.plan.diagnostics
        if c.metrics.get("crossings", 0):
            out.append(f"网络有 {c.metrics['crossings']} 处交叉：请重新生成或减少节点数")
        if c.metrics.get("edges_outside", 0):
            out.append(f"{c.metrics['edges_outside']} 段走廊跨出轮廓凹角（L / U 型平面）：请重新生成")
        if d.get("n_vertical_cores", 0) > 3:
            out.append(f"{d['n_vertical_cores']} 个内部断头被解释为竖向交通核；若为首层请检查是否应为出入口")
        if (c.metrics.get("sharp_angle_rate") or 0) > 0.25:
            out.append("锐角比例偏高（真实 ≈ 0.13）：商铺进深在楔形处不足")
        if d.get("corridor_ratio", 0) > 0.28:
            out.append("走廊占比超过真实分布上四分位（0.28）：可租面积偏低")
        if (c.indicators.get("max_betweenness") or 0) > 0.5:
            out.append("最大介数 > 0.5：人流过度集中在一段走廊")
        return out

    # ---- export ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def export_candidate(c: Candidate, out_dir: str | Path, stem: str = "candidate", formats: tuple[str, ...] = ("json", "geojson", "svg", "png")) -> dict[str, Path]:
        from mall_space_planner.hub.export import candidate_to_geojson, candidate_to_json, candidate_to_png, candidate_to_svg

        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if "json" in formats:
            paths["json"] = out / f"{stem}.json"; paths["json"].write_text(json.dumps(candidate_to_json(c), ensure_ascii=False, indent=1), encoding="utf-8")
        if "geojson" in formats:
            paths["geojson"] = out / f"{stem}.geojson"; paths["geojson"].write_text(json.dumps(candidate_to_geojson(c)), encoding="utf-8")
        if "svg" in formats:
            paths["svg"] = out / f"{stem}.svg"; paths["svg"].write_text(candidate_to_svg(c), encoding="utf-8")
        if "png" in formats:
            paths["png"] = candidate_to_png(c, out / f"{stem}.png")
        return paths


__all__ = ["Candidate", "Workbench"]
