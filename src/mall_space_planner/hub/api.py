"""FastAPI backend of the Viewer Hub.

All business logic lives in ``mall_space_planner.hub.*`` (Catalog / Workbench / ExperimentRegistry / JobRunner); this
module only maps HTTP routes onto those objects. The Streamlit UI calls the same objects in-process; a later React front-end
can call these routes instead without touching the algorithms.

Run:  ``uvicorn mall_space_planner.hub.api:app --port 8000``  (PYTHONPATH=src)
"""
from __future__ import annotations

import dataclasses
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from mall_space_planner.hub.catalog import Catalog  # noqa: E402
from mall_space_planner.hub.experiments import ExperimentRegistry  # noqa: E402
from mall_space_planner.hub.export import candidate_to_geojson, candidate_to_json, candidate_to_png, candidate_to_svg, draw_candidate  # noqa: E402
from mall_space_planner.hub.jobs import JobRunner, JobSpec  # noqa: E402
from mall_space_planner.hub.workbench import Candidate, Workbench  # noqa: E402
from mall_space_planner.schemas import PlanningCondition  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
app = FastAPI(title="MallTopoCehua Viewer Hub API", version="0.5.0")


class _State:
    catalog: Catalog | None = None
    workbench: Workbench | None = None
    registry: ExperimentRegistry | None = None
    jobs: JobRunner | None = None
    sessions: dict[str, dict[str, Any]] = {}  # session_id -> {"candidates": [Candidate], "request": {...}}
    renovations: dict[str, dict[str, Any]] = {}


S = _State()


def catalog() -> Catalog:
    if S.catalog is None:
        S.catalog = Catalog.load()
    return S.catalog


def workbench() -> Workbench:
    if S.workbench is None:
        S.workbench = Workbench.load(catalog())
    return S.workbench


def registry() -> ExperimentRegistry:
    if S.registry is None:
        S.registry = ExperimentRegistry()
    return S.registry.scan()


def jobs() -> JobRunner:
    if S.jobs is None:
        S.jobs = JobRunner()
    return S.jobs


def _df_records(df) -> list[dict[str, Any]]:  # noqa: ANN001
    return json.loads(df.to_json(orient="records", force_ascii=False))


# ------------------------------------------------------------------------------------------------------- catalog (UI 1)
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "time": time.time()}


@app.get("/api/catalog/overview")
def catalog_overview() -> dict[str, Any]:
    return json.loads(json.dumps(catalog().overview(), default=str))


@app.get("/api/catalog/missing")
def catalog_missing() -> list[dict[str, Any]]:
    return _df_records(catalog().missing_report())


@app.get("/api/catalog/anomalies")
def catalog_anomalies() -> list[dict[str, Any]]:
    return _df_records(catalog().anomalies())


class FilterRequest(BaseModel):
    layout_types: list[str] | None = None
    clusters: list[int] | None = None
    score_range: tuple[float, float] | None = None
    area_range: tuple[float, float] | None = None
    splits: list[str] | None = None
    has_graph: bool | None = None
    text: str | None = None
    limit: int = 500


@app.post("/api/catalog/filter")
def catalog_filter(req: FilterRequest) -> list[dict[str, Any]]:
    df = catalog().filter(req.layout_types, req.clusters, req.score_range, req.area_range, req.splits, req.has_graph, req.text)
    return _df_records(df.head(req.limit))


@app.get("/api/catalog/case/{floor_id}")
def catalog_case(floor_id: str) -> dict[str, Any]:
    c = catalog()
    try:
        row = c.case(floor_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, str(e)) from e
    topo = c.topology(floor_id)
    return json.loads(json.dumps({"case": row, "topology": topo.model_dump() if topo is not None else None, "metrics": c.topology_metrics(floor_id)}, default=str))


@app.post("/api/catalog/compare")
def catalog_compare(ids: list[str]) -> dict[str, Any]:
    df = catalog().compare(ids)
    return {"columns": list(map(str, df.columns)), "rows": json.loads(df.to_json(orient="split", force_ascii=False))}


# --------------------------------------------------------------------------------------------------- planning (UI 3)
class ConditionRequest(BaseModel):
    condition: dict[str, Any]
    layout_type: str | None = None
    top_k: int = 5


@app.get("/api/plan/default_condition")
def default_condition(cluster: int = 2) -> dict[str, Any]:
    return workbench().default_condition(cluster).model_dump()


@app.post("/api/plan/recommend_types")
def recommend_types(req: ConditionRequest) -> dict[str, Any]:
    r = workbench().recommend_types(PlanningCondition(**req.condition))
    return json.loads(json.dumps(dataclasses.asdict(r), default=str))


@app.post("/api/plan/recommend")
def recommend(req: ConditionRequest) -> list[dict[str, Any]]:
    recs = workbench().recommend(PlanningCondition(**req.condition), req.layout_type, req.top_k)
    return [r.model_dump(warnings=False) for r in recs]


class GenerateRequest(BaseModel):
    prototype_id: str
    condition: dict[str, Any] | None = None
    outline_points: list[tuple[float, float]] | None = None
    outline_floor_id: str | None = None
    outline_area_m2: float | None = None
    n_target: int = 30
    n_candidates: int = 3
    seed: int = 0
    layout_type: str | None = None
    target_avg_degree: float | None = None
    corridor_ratio: float | None = None
    min_entrances: int = 2
    max_entrances: int = 6
    max_atria: int = 2
    restarts: int = 4
    iters: int = 100


def _cand_payload(c: Candidate) -> dict[str, Any]:
    wb = workbench()
    return json.loads(json.dumps({"summary": c.summary(), "constraints": wb.constraint_report(c), "risks": wb.risks(c), "metrics": c.metrics, "indicators": c.indicators, "diagnostics": c.plan.diagnostics}, default=str))


@app.post("/api/plan/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    wb = workbench()
    sk = catalog().topology(req.prototype_id)
    if sk is None:
        raise HTTPException(404, f"prototype {req.prototype_id} has no topology")
    outline, src = wb.outline_from(req.outline_points, req.outline_floor_id, req.outline_area_m2)
    cond = PlanningCondition(**req.condition) if req.condition else None
    kw = dict(layout_type=req.layout_type, target_avg_degree=req.target_avg_degree, corridor_ratio=req.corridor_ratio, min_entrances=req.min_entrances, max_entrances=req.max_entrances, max_atria=req.max_atria,
              score_fn=wb.score_fn(cond) if cond else None, restarts=req.restarts, iters=req.iters)
    cands = wb.generate_candidates(sk, req.prototype_id, outline, req.n_target, req.n_candidates, req.seed, **kw)
    sid = uuid.uuid4().hex[:10]
    S.sessions[sid] = {"candidates": cands, "request": req.model_dump(), "kw": kw, "outline": outline, "skeleton": sk}
    return {"session_id": sid, "outline_source": src, "candidates": [_cand_payload(c) for c in cands]}


class RegenerateRequest(BaseModel):
    session_id: str
    index: int
    seed: int | None = None


@app.post("/api/plan/regenerate")
def regenerate(req: RegenerateRequest) -> dict[str, Any]:
    """One-click regenerate of a single unsatisfying candidate: same prototype/outline/constraints, new seed."""
    s = S.sessions.get(req.session_id)
    if s is None:
        raise HTTPException(404, "unknown session")
    if not 0 <= req.index < len(s["candidates"]):
        raise HTTPException(400, "bad candidate index")
    wb = workbench()
    r = s["request"]
    old = s["candidates"][req.index]
    seed = req.seed if req.seed is not None else old.seed + 1000 + int(time.time()) % 997
    c = wb.generate_candidate(s["skeleton"], r["prototype_id"], s["outline"], r["n_target"], seed, index=req.index, **s["kw"])
    s["candidates"][req.index] = c
    return {"session_id": req.session_id, "index": req.index, "candidate": _cand_payload(c)}


@app.get("/api/plan/{session_id}/candidate/{index}.{fmt}")
def candidate_file(session_id: str, index: int, fmt: str) -> Response:
    s = S.sessions.get(session_id)
    if s is None or not 0 <= index < len(s["candidates"]):
        raise HTTPException(404, "unknown session / candidate")
    c = s["candidates"][index]
    if fmt == "json":
        return JSONResponse(candidate_to_json(c))
    if fmt == "geojson":
        return JSONResponse(candidate_to_geojson(c), media_type="application/geo+json")
    if fmt == "svg":
        return Response(candidate_to_svg(c), media_type="image/svg+xml")
    if fmt == "png":
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_candidate(ax, c, title=f"candidate {index + 1} (seed {c.seed})")
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return Response(buf.getvalue(), media_type="image/png")
    raise HTTPException(400, "fmt must be json|geojson|svg|png")


# --------------------------------------------------------------------------------------------------- renovation (UI 3)
@app.get("/api/renovation/floors")
def renovation_floors(low_score: float | None = None, limit: int = 50, min_nodes: int = 12, min_area: float = 6000.0) -> list[str]:
    return workbench().renovation_floors(low_score, limit, min_nodes, min_area)


class RenovateRequest(BaseModel):
    floor_id: str
    seed: int = 0
    n_candidates: int = 6
    use_score: bool = True
    keep_skeleton_positions: bool = False
    restarts: int = 4
    iters: int = 100


@app.post("/api/renovation/run")
def renovate(req: RenovateRequest) -> dict[str, Any]:
    r = workbench().renovate(req.floor_id, req.seed, req.n_candidates, req.use_score, req.keep_skeleton_positions, req.restarts, req.iters)
    rid = uuid.uuid4().hex[:10]
    S.renovations[rid] = r
    return json.loads(json.dumps({"renovation_id": rid, "floor_id": req.floor_id, "seed": req.seed, "generator": r["generator_name"], "table": r["table"], "row": r["row"], "anchored": sorted(r["anchored"]), "n_entrance_points": len(r["entrance_points"])}, default=str))


@app.get("/api/renovation/{rid}.png")
def renovation_png(rid: str) -> Response:
    r = S.renovations.get(rid)
    if r is None:
        raise HTTPException(404, "unknown renovation")
    from mall_space_planner.hub.viz import draw_renovation

    fig = draw_renovation(r)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Response(buf.getvalue(), media_type="image/png")


# --------------------------------------------------------------------------------------------------- jobs (UI 2)
class JobRequest(BaseModel):
    kind: str
    args: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


@app.get("/api/jobs")
def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    return [j.to_dict() for j in jobs().list(limit)]


@app.post("/api/jobs")
def submit_job(req: JobRequest) -> dict[str, Any]:
    try:
        return jobs().submit(JobSpec(req.kind, req.args, req.label)).to_dict()
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"bad job spec: {e}") from e


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    j = jobs().refresh(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j.to_dict()


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, n: int = 200) -> str:
    return jobs().tail(job_id, n)


@app.get("/api/jobs/{job_id}/curves")
def job_curves(job_id: str) -> dict[str, list[float]]:
    return jobs().parse_curves(job_id)


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str) -> dict[str, Any]:
    ok = jobs().stop(job_id)
    j = jobs().refresh(job_id)
    return {"stopped": bool(ok), "status": j.status if j else "unknown"}


# --------------------------------------------------------------------------------------------------- experiments (UI 2)
@app.get("/api/experiments")
def experiments(family: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    return _df_records(registry().table(family, split))


@app.get("/api/experiments/families")
def experiment_families() -> list[str]:
    return registry().families()


@app.post("/api/experiments/compare")
def experiments_compare(names: list[str], split: str = "test") -> dict[str, Any]:
    df = registry().compare(names, split)
    return json.loads(df.to_json(orient="split", force_ascii=False)) if not df.empty else {}


@app.get("/api/experiments/curves")
def experiment_curves(name: str) -> dict[str, list[float]]:
    return registry().curves(name)


@app.get("/api/experiments/checkpoints")
def experiment_checkpoints() -> list[dict[str, Any]]:
    return _df_records(registry().checkpoints())


@app.get("/api/experiments/export")
def experiments_export(family: str | None = None, split: str | None = None, fmt: str = "csv") -> Response:
    df = registry().table(family, split)
    if fmt == "csv":
        return Response(df.to_csv(index=False), media_type="text/csv")
    if fmt == "md":
        return PlainTextResponse(ExperimentRegistry.to_markdown(df))
    if fmt == "tex":
        return PlainTextResponse(ExperimentRegistry.to_latex(df, caption=f"{family or 'all'} ({split or 'all splits'})", label=f"tab:{family or 'all'}"))
    raise HTTPException(400, "fmt must be csv|md|tex")


__all__ = ["app", "candidate_to_png"]
