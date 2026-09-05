"""UI-agnostic planning service used by CLI, Streamlit and (later) FastAPI.

Wraps: load processed DB → load/fit Stage-1 → recommend → build Stage-2 request → generate
→ export. No Streamlit imports here; everything returns plain schema objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.geometry.export import layout_to_geojson, layout_to_json, layout_to_svg
from mall_space_planner.schemas import ConstraintSet, EvaluationResult, GeneratedLayout, PlanningCondition, Recommendation, SiteBoundary
from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
from mall_space_planner.stage1.type_recommender import TreeTypeRecommender, TypeRecommenderResult
from mall_space_planner.stage2.base import GenerationRequest
from mall_space_planner.stage2.pipelines.generate import Stage2Pipeline
from mall_space_planner.utils.logging import get_logger
from mall_space_planner.visualization.render import render_layout_png

logger = get_logger(__name__)


class PlanningService:
    def __init__(self, db: CaseDatabase, stage1_cfg: dict[str, Any], stage2_cfg: dict[str, Any], checkpoint_dir: str | Path | None = None) -> None:
        self.db = db
        self.stage1_cfg = stage1_cfg
        self.stage2_cfg = stage2_cfg
        ck = Path(checkpoint_dir) if checkpoint_dir else None
        if ck and (ck / "checkpoint_meta.json").exists():
            logger.info("Loading Stage-1 checkpoint from %s", ck)
            self.stage1 = Stage1Pipeline.load(ck, db, stage1_cfg)
        else:
            logger.info("No checkpoint given/found; fitting Stage-1 pipeline in-process")
            self.stage1 = Stage1Pipeline(stage1_cfg, db).fit()
        self.stage2 = Stage2Pipeline(stage2_cfg)
        tcfg = stage1_cfg.get("stage1", {}).get("type_recommender", {"enabled": True})
        self.type_rec: TreeTypeRecommender | None = None
        if tcfg and tcfg.get("enabled", True):
            self.type_rec = TreeTypeRecommender(**{k: v for k, v in tcfg.items() if k != "enabled"}).fit(db, db.split("train"))

    # ------------------------------------------------------------------ layout type (design decision support)
    def recommend_types(self, condition: PlanningCondition) -> TypeRecommenderResult | None:
        """Expected quality per layout type under the given conditions (E[score | conditions, type])."""
        return self.type_rec.recommend(self.db, condition) if self.type_rec is not None else None

    def recommend_within_type(self, condition: PlanningCondition, layout_type: str, top_k: int = 10, **kw: Any) -> list[Recommendation]:
        """Quality-aware comparable-case retrieval restricted to the user-selected layout type."""
        from mall_space_planner.schemas import LayoutType

        q = condition.model_copy(update={"preferred_layout": LayoutType(layout_type)})
        return self.stage1.recommend(q, top_k=top_k, **kw)

    # ------------------------------------------------------------------ stage 1
    def recommend(self, condition: PlanningCondition, top_k: int = 10, with_counterfactuals: bool = True) -> list[Recommendation]:
        return self.stage1.recommend(condition, top_k=top_k, with_counterfactuals=with_counterfactuals)

    def prototype(self, prototype_id: str):  # noqa: ANN201
        case = self.db.get_case(prototype_id)
        if case.prototype is None:
            raise KeyError(f"prototype {prototype_id} has no graph")
        return case.prototype

    # ------------------------------------------------------------------ stage 2
    def generate(self, prototype_id: str, boundary: SiteBoundary, constraints: ConstraintSet, n_candidates: int = 3, seed: int = 0) -> list[tuple[GeneratedLayout, EvaluationResult]]:
        req = GenerationRequest(prototype=self.prototype(prototype_id), boundary=boundary, constraints=constraints, n_candidates=n_candidates, seed=seed)
        return self.stage2.run(req)

    @staticmethod
    def export(layout: GeneratedLayout, out_dir: str | Path, stem: str = "layout", formats: tuple[str, ...] = ("json", "geojson", "svg", "png")) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if "json" in formats:
            layout_to_json(layout, out_dir / f"{stem}.json"); paths["json"] = out_dir / f"{stem}.json"
        if "geojson" in formats:
            layout_to_geojson(layout, out_dir / f"{stem}.geojson"); paths["geojson"] = out_dir / f"{stem}.geojson"
        if "svg" in formats:
            layout_to_svg(layout, out_dir / f"{stem}.svg"); paths["svg"] = out_dir / f"{stem}.svg"
        if "png" in formats:
            render_layout_png(layout, out_dir / f"{stem}.png"); paths["png"] = out_dir / f"{stem}.png"
        return paths

    @staticmethod
    def recommendations_to_json(recs: list[Recommendation], path: str | Path) -> Path:
        Path(path).write_text(json.dumps([r.model_dump(mode="json") for r in recs], ensure_ascii=False, indent=2), encoding="utf-8")
        return Path(path)
