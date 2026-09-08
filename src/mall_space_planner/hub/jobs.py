"""Background job runner for long tasks (training, evaluation, ablations, thesis figures).

Each job is a subprocess started with ``Popen`` in its own process group, writing stdout/stderr to
``<jobs_dir>/<job_id>/log.txt`` and its metadata to ``meta.json``. Status is derived from the process (running) or the
recorded return code (finished / failed / stopped), so the registry survives a UI restart: the Streamlit process may be
reloaded while the training keeps running.

The runner knows the project's scripts and builds their command lines (``JobSpec``), so the UI never assembles shell
commands itself.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOBS_DIR = ROOT / "outputs" / "jobs"


@dataclass
class JobSpec:
    kind: str  # train_stage1 | train_stage2 | evaluate_stage1 | evaluate_stage2 | evaluate_stage3 | ablation | renovate | floor_plates | figures | custom
    args: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def command(self) -> list[str]:
        py = sys.executable
        a = self.args
        ov = [f"{k}={v}" for k, v in (a.get("override") or a.get("overrides") or {}).items()]
        if self.kind == "train_stage1":
            cmd = [py, "scripts/train_stage1.py", "--config", a["config"]]
            if a.get("seed") is not None:
                cmd += ["--seed", str(a["seed"])]
        elif self.kind == "train_stage2":
            cmd = [py, "scripts/train_stage2.py", "--config", a["config"]]
            if a.get("corpus"):
                cmd += ["--corpus", a["corpus"]]
            if a.get("limit"):
                cmd += ["--limit", str(a["limit"])]
            if a.get("force"):
                cmd += ["--force"]
        elif self.kind == "evaluate_stage1":
            cmd = [py, "scripts/evaluate_stage1.py", "--config", a["config"], "--checkpoint", a["checkpoint"]]
        elif self.kind == "evaluate_stage2":
            cmd = [py, "scripts/evaluate_stage2.py", "--config", a["config"], "--limit", str(a.get("limit", 200)), "--seed", str(a.get("seed", 0))]
            if a.get("corpus"):
                cmd += ["--corpus", a["corpus"]]
            if a.get("ground_truth"):
                cmd += ["--ground-truth"]
            if a.get("force"):
                cmd += ["--force"]
        elif self.kind == "evaluate_stage3":
            cmd = [py, "scripts/evaluate_stage3.py", "--config", a["config"], "--split", a.get("split", "test"), "--limit", str(a.get("limit", 50)), "--out", a.get("out", "outputs/experiments/stage3_eval")]
            if a.get("resume"):
                cmd += ["--resume"]
        elif self.kind == "renovate":
            cmd = [py, "scripts/renovate_stage3.py", "--config", a["config"], "--out", a.get("out", "outputs/experiments/renovation")]
            if a.get("floors"):
                cmd += ["--floors", *a["floors"]]
            if a.get("split"):
                cmd += ["--split", a["split"]]
            if a.get("limit"):
                cmd += ["--limit", str(a["limit"])]
            if a.get("low_score") is not None:
                cmd += ["--low-score", str(a["low_score"])]
            if a.get("no_filter"):
                cmd += ["--no-filter"]
        elif self.kind == "ablation":
            cmd = [py, "scripts/run_ablation.py", "--config", a["config"]]
            if a.get("out_dir"):
                cmd += ["--out-dir", a["out_dir"]]
        elif self.kind == "floor_plates":
            cmd = [py, "scripts/export_floor_plates.py", "--out", a.get("out", "outputs/floor_plates"), "--workers", str(a.get("workers", 4))]
            if a.get("config"):
                cmd += ["--config", a["config"]]
            if a.get("force"):
                cmd += ["--force"]
            if a.get("limit"):
                cmd += ["--limit", str(a["limit"])]
        elif self.kind == "figures":
            cmd = [py, "scripts/make_thesis_report.py"]
            if a.get("only"):
                cmd += ["--only", *a["only"]]
        elif self.kind == "custom":
            c = a.get("command") or a.get("cmd")
            if not c:
                raise ValueError("custom job needs args['command'] (string or list)")
            cmd = list(c) if isinstance(c, (list, tuple)) else shlex.split(str(c))
        else:
            raise ValueError(f"unknown job kind {self.kind}")
        if ov and self.kind in {"train_stage1", "train_stage2", "evaluate_stage1", "evaluate_stage2", "ablation"}:
            cmd += ["--override", *ov]
        return cmd


@dataclass
class JobInfo:
    job_id: str
    kind: str
    label: str
    command: list[str]
    started_at: float
    pid: int | None = None
    finished_at: float | None = None
    returncode: int | None = None
    stopped: bool = False
    workdir: str = str(ROOT)

    @property
    def status(self) -> str:
        if self.stopped:
            return "stopped"
        if self.returncode is None:
            return "running" if self.pid and _alive(self.pid) else "unknown"
        return "finished" if self.returncode == 0 else "failed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        d["elapsed_s"] = round((self.finished_at or time.time()) - self.started_at, 1)
        return d


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # zombie check
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line
    except OSError:
        pass
    return True


class JobRunner:
    def __init__(self, jobs_dir: str | Path = DEFAULT_JOBS_DIR) -> None:
        self.dir = Path(jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}

    # ---- persistence ------------------------------------------------------------------------------------------------
    def _meta_path(self, job_id: str) -> Path:
        return self.dir / job_id / "meta.json"

    def _save(self, info: JobInfo) -> None:
        self._meta_path(info.job_id).parent.mkdir(parents=True, exist_ok=True)
        self._meta_path(info.job_id).write_text(json.dumps(asdict(info), indent=1), encoding="utf-8")

    def _load(self, job_id: str) -> JobInfo | None:
        p = self._meta_path(job_id)
        if not p.exists():
            return None
        return JobInfo(**json.loads(p.read_text(encoding="utf-8")))

    def log_path(self, job_id: str) -> Path:
        return self.dir / job_id / "log.txt"

    # ---- lifecycle --------------------------------------------------------------------------------------------------
    def submit(self, spec: JobSpec, env: dict[str, str] | None = None) -> JobInfo:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        cmd = spec.command()
        (self.dir / job_id).mkdir(parents=True, exist_ok=True)
        log = open(self.log_path(job_id), "w", encoding="utf-8")  # noqa: SIM115 (closed by the child)
        log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        log.flush()
        e = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), **(env or {})}
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=e, start_new_session=True)
        self._procs[job_id] = proc
        info = JobInfo(job_id=job_id, kind=spec.kind, label=spec.label or spec.kind, command=cmd, started_at=time.time(), pid=proc.pid)
        self._save(info)
        return info

    def refresh(self, job_id: str) -> JobInfo | None:
        info = self._load(job_id)
        if info is None:
            return None
        if info.returncode is None and not info.stopped:
            proc = self._procs.get(job_id)
            if proc is not None:
                rc = proc.poll()
                if rc is not None:
                    info.returncode, info.finished_at = rc, time.time()
                    self._save(info)
            elif info.pid and not _alive(info.pid):
                # started by another process (e.g. a previous UI session) and already gone: rc unknown → mark by log tail
                tail = self.tail(job_id, 20).lower()
                info.returncode, info.finished_at = (1 if ("traceback" in tail or "error" in tail) else 0), time.time()
                self._save(info)
        return info

    def stop(self, job_id: str) -> JobInfo | None:
        info = self.refresh(job_id)
        if info is None or info.status != "running":
            return info
        try:
            os.killpg(os.getpgid(info.pid), signal.SIGTERM)  # type: ignore[arg-type]
            for _ in range(20):
                if not _alive(info.pid):  # type: ignore[arg-type]
                    break
                time.sleep(0.1)
            if _alive(info.pid):  # type: ignore[arg-type]
                os.killpg(os.getpgid(info.pid), signal.SIGKILL)  # type: ignore[arg-type]
        except ProcessLookupError:
            pass
        info.stopped, info.finished_at = True, time.time()
        self._save(info)
        return info

    def list(self, limit: int = 50) -> list[JobInfo]:
        ids = sorted((p.name for p in self.dir.iterdir() if (p / "meta.json").exists()), reverse=True)[:limit]
        return [i for i in (self.refresh(j) for j in ids) if i is not None]

    def tail(self, job_id: str, n_lines: int = 200) -> str:
        p = self.log_path(job_id)
        if not p.exists():
            return ""
        try:
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 200_000))
                data = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(data.splitlines()[-n_lines:])

    def parse_curves(self, job_id: str) -> dict[str, list[float]]:
        """Training curves from the log: lines like ``ARGNN epoch 3/10: loss=4.01 val_anchor_acc=0.374 ...`` or any
        ``key=value`` numeric pairs following an ``epoch`` token."""
        import re

        out: dict[str, list[float]] = {}
        for line in self.tail(job_id, 100_000).splitlines():
            if "epoch" not in line.lower():
                continue
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?(?:e-?\d+)?)", line):
                out.setdefault(k, []).append(float(v))
        return out


__all__ = ["DEFAULT_JOBS_DIR", "JobInfo", "JobRunner", "JobSpec"]
