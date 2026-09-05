"""Evidence-based template explainer.

No language model is involved. The explanation is assembled from:

1. feature importance of the ranker (global) — *which inputs drive the ranking*;
2. per-candidate condition differences (standardised) — *how similar is this case*;
3. topology metric position within the candidate pool (percentiles) — *what kind of
   skeleton is this*;
4. hard-filter / relaxation log — *which constraints were applied*;
5. counterfactuals produced by the pipeline (re-scoring under perturbed conditions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.registry import register
from mall_space_planner.schemas import PlanningCondition, RecommendationExplanation
from mall_space_planner.stage1.base import BaseExplainer, RankingContext

_FEATURE_LABELS = {
    "people": "城市人口",
    "GDP_2023": "城市GDP",
    "PCDI_2023": "人均可支配收入",
    "TP_2023": "消费总量指标",
    "mall_area_count": "区域商场数量",
    "nearest_distance_km": "最近商场距离",
    "count_1km": "1km内商场数",
    "count_2km": "2km内商场数",
    "total_area": "项目总面积",
    "Tx": "Tx 指标",
    "L1_density": "路网密度(L1)",
    "L2_diameter": "拓扑直径(L2)",
    "L2_complexity": "拓扑复杂度(L2)",
    "L2_integration": "整合度(L2)",
    "g_density": "图密度",
    "g_avg_shortest_path": "平均最短路径",
    "g_num_cycles": "环路数",
    "g_num_nodes": "节点数",
    "g_avg_degree": "平均度",
}


def _label(name: str) -> str:
    base = name.split(":", 1)[-1]
    return _FEATURE_LABELS.get(base, base)


@register("explainer", "template")
class TemplateExplainer(BaseExplainer):
    def __init__(self, top_factors: int = 5, top_metrics: int = 4) -> None:
        self.top_factors = top_factors
        self.top_metrics = top_metrics

    def explain(
        self,
        ctx: RankingContext,
        query: PlanningCondition,
        ranked: pd.DataFrame,
        rank_index: int,
        model_evidence: dict[str, Any] | None = None,
    ) -> RecommendationExplanation:
        ev = model_evidence or {}
        row = ranked.iloc[rank_index]
        spec = ctx.features.spec
        pool = ctx.candidates

        # 1) global factors from the ranker
        importance: dict[str, float] = ev.get("feature_importance") or {}
        factors = sorted(importance.items(), key=lambda kv: -abs(kv[1]))[: self.top_factors]
        top_factors = [{"feature": k, "label": _label(k), "importance": float(v)} for k, v in factors]

        # 2) similarity evidence: standardised condition differences
        q_df = pd.DataFrame([{c: getattr(query, c, None) for c in spec.query_cols}])
        q = ctx.features.condition_matrix(q_df)[0]
        c = ctx.features.condition_matrix(ranked.iloc[[rank_index]])[0]
        diffs = sorted(zip(spec.query_cols, (c - q).tolist(), strict=True), key=lambda kv: abs(kv[1]))
        closest = [{"feature": k, "label": _label(k), "std_diff": round(v, 3)} for k, v in diffs[:3]]
        farthest = [{"feature": k, "label": _label(k), "std_diff": round(v, 3)} for k, v in diffs[-2:]]

        # 3) topology metrics relative to the pool
        topo_lines: list[str] = []
        metric_cols = [m for m in (spec.metric_cols + spec.graph_metric_cols) if m in pool.columns][: self.top_metrics + 4]
        for m in metric_cols:
            val = row.get(m)
            if val is None or pd.isna(val):
                continue
            pct = float((pool[m] < val).mean() * 100)
            topo_lines.append(f"{_label(m)} = {float(val):.3f}（位于候选池第 {pct:.0f} 百分位）")
        topo_lines = topo_lines[: self.top_metrics]

        # 4) constraints
        applied = ev.get("constraints_applied", [])
        relaxed = ev.get("constraints_relaxed", [])

        # 5) risks
        risks: list[str] = []
        if relaxed:
            risks.append("部分硬约束被放宽：" + "；".join(relaxed))
        if farthest and abs(farthest[-1]["std_diff"]) > 1.5:
            risks.append(f"候选案例在“{farthest[-1]['label']}”上与输入条件差异较大（{farthest[-1]['std_diff']:+.2f} σ）")
        if "g_n_components" in row and row.get("g_n_components", 1) and row["g_n_components"] > 1:
            risks.append("该原型骨架本身不连通，需在阶段二修复")
        qs = row.get(ctx.db.label_col)
        if qs is not None and not pd.isna(qs) and ctx.db.label_col in pool:
            pct_q = float((pool[ctx.db.label_col] < qs).mean() * 100)
            if pct_q < 50:
                risks.append(f"案例质量评分仅位于第 {pct_q:.0f} 百分位")

        conf = ev.get("confidence")
        # framing: "quality-aware comparable case within the (user-)selected layout type"
        lt = row.get("layout_type")
        lt_txt = "" if lt is None or pd.isna(lt) else f"，布局类型「{lt}」"
        pref = getattr(query, "preferred_layout", None)
        type_txt = f"（已按所选类型「{getattr(pref, 'value', pref)}」限定检索范围）" if pref is not None else ""
        qual_txt = ""
        if qs is not None and not pd.isna(qs) and ctx.db.label_col in pool:
            pct_q = float((pool[ctx.db.label_col] < qs).mean() * 100)
            qual_txt = f"该案例评分 {float(qs):.2f}，位于同条件可比案例的第 {pct_q:.0f} 百分位。"
        summary = (
            f"可比案例 {row[ctx.db.id_col]}{lt_txt}（第 {rank_index + 1} 名，模型分数 {float(row['score']):.3f}）{type_txt}。"
            f"可比范围：{'、'.join(applied) if applied else '无硬约束'}。"
            f"与输入策划条件最接近的因素：{'、'.join(x['label'] for x in closest)}。{qual_txt}"
        )
        return RecommendationExplanation(
            recommendation_summary=summary,
            top_factors=top_factors,
            matched_case_evidence=[
                {"case_id": str(row[ctx.db.id_col]), "mall_id": str(row.get(ctx.db.mall_id_col)), "quality_score": None if pd.isna(qs) else float(qs), "similarity": None if "similarity" not in row or pd.isna(row["similarity"]) else float(row["similarity"]), "closest_conditions": closest, "largest_gaps": farthest}
            ],
            topology_reasoning=topo_lines,
            risks=risks,
            counterfactuals=ev.get("counterfactuals", []),
            confidence=None if conf is None else float(np.clip(conf, 0, 1)),
        )
