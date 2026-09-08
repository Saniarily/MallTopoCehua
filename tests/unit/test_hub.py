"""Viewer Hub backend: catalog, experiment registry, job runner, FastAPI routes (fast paths only)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog():
    from mall_space_planner.hub.catalog import Catalog

    return Catalog.load()


def test_catalog_overview_filter_case(catalog):
    ov = catalog.overview()
    assert ov["n_floors"] > 0 and ov["n_graphs"] > 0
    test = catalog.filter(splits=["test"])
    assert len(test) > 0 and (test["split"] == "test").all()
    fid = str(test.iloc[0][catalog.db.id_col])
    row = catalog.case(fid)
    assert row[catalog.db.id_col] == fid
    m = catalog.topology_metrics(fid)
    assert m["num_nodes"] > 0
    cmp = catalog.compare([fid, str(test.iloc[1][catalog.db.id_col])])
    assert cmp.shape[1] == 2


def test_catalog_anomalies_scale_aware(catalog):
    an = catalog.anomalies()
    # synthetic scores live in [0, 100]; the range check must not flag every row
    assert not ((an["type"] == "score_out_of_range") & (an["n"] == len(catalog.cases))).any()


def test_experiment_registry_scans_and_exports():
    from mall_space_planner.hub.experiments import ExperimentRegistry

    reg = ExperimentRegistry().scan()
    tbl = reg.table()
    assert {"name", "family", "split"} <= set(tbl.columns)
    md = ExperimentRegistry.to_markdown(tbl.head(3))
    assert md.startswith("| name")
    tex = ExperimentRegistry.to_latex(tbl.head(2))
    assert "\\begin{tabular}" in tex


def test_job_runner_lifecycle(tmp_path):
    from mall_space_planner.hub.jobs import JobRunner, JobSpec

    jr = JobRunner(tmp_path)
    script = "import time\nfor e in range(30):\n    print(f'epoch={e} loss={1/(e+1):.3f}', flush=True); time.sleep(0.2)"
    j = jr.submit(JobSpec("custom", {"command": [sys.executable, "-u", "-c", script]}, "demo"))
    time.sleep(1.0)
    assert jr.refresh(j.job_id).status == "running"
    assert "epoch=" in jr.tail(j.job_id, 5)
    assert "loss" in jr.parse_curves(j.job_id)
    assert jr.stop(j.job_id)
    time.sleep(0.5)
    assert jr.refresh(j.job_id).status == "stopped"
    j2 = jr.submit(JobSpec("custom", {"command": f"{sys.executable} -c \"raise SystemExit(3)\""}, "fail"))
    time.sleep(1.0)
    assert jr.refresh(j2.job_id).status == "failed"
    assert {x.job_id for x in jr.list()} == {j.job_id, j2.job_id}


def test_job_spec_commands():
    from mall_space_planner.hub.jobs import JobSpec

    cmd = JobSpec("train_stage1", {"config": "configs/stage1/extra_trees.yaml", "seed": 1, "override": {"a.b": 2}}).command()
    assert cmd[1:] == ["scripts/train_stage1.py", "--config", "configs/stage1/extra_trees.yaml", "--seed", "1", "--override", "a.b=2"]
    with pytest.raises(ValueError):
        JobSpec("nope", {}).command()


def test_api_fast_routes():
    from fastapi.testclient import TestClient

    from mall_space_planner.hub.api import app

    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    assert c.get("/api/catalog/overview").json()["n_floors"] > 0
    rows = c.post("/api/catalog/filter", json={"splits": ["test"], "limit": 3}).json()
    assert len(rows) == 3
    assert c.get(f"/api/catalog/case/{rows[0]['floor_id']}").status_code == 200
    assert c.get("/api/catalog/case/NOPE").status_code == 404
    assert c.post("/api/jobs", json={"kind": "train_stage1", "args": {}}).status_code == 400
    assert isinstance(c.get("/api/experiments/families").json(), list)
    assert c.get("/api/experiments/export?fmt=md").status_code == 200


def test_export_floor_plates_fixture(tmp_path):
    from mall_space_planner.hub.plates import find_prerendered_thumbnail, load_plates_table, render_floor_plates
    from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths

    ds = Stage3Dataset(Stage3Paths(graph_dir=ROOT / "tests/fixtures/graph_csv"))
    row = render_floor_plates("B000A0E928_1", ds, tmp_path)
    assert row["status"] == "ok" and row["n_nodes"] == 50 and row["nodes_outside"] == 0
    for k in ("plan_topo", "outline_topo", "thumbs"):
        assert (tmp_path / k / "B000A0E928_1.png").stat().st_size > 1000
    assert render_floor_plates("B000A0E928_1", ds, tmp_path)["status"] == "exists"
    import pandas as pd

    pd.DataFrame([row]).to_csv(tmp_path / "floors.csv", index=False)
    assert find_prerendered_thumbnail("B000A0E928_1", tmp_path) is not None
    assert load_plates_table(tmp_path)["floor_id"].iloc[0] == "B000A0E928_1"
