"""Experiment registry for the Viewer Hub (UI 2).

Scans the result folders written by the project's scripts and normalises them into a flat table so the UI (or the API)
can list, compare and export experiments without knowing the on-disk layout:

* ``outputs/experiments/stage1/<name>/{test,val}_metrics.json`` + ``checkpoint/checkpoint_meta.json``  (train_stage1)
* ``outputs/experiments/<ablation>/<variant>/seed_*/run.json``                                          (run_ablation)
* ``outputs/experiments/stage2_eval/<name>/aggregate.json`` + ``per_sample.csv``                        (evaluate_stage2)
* ``outputs/experiments/**/summary.json``                                                                (evaluate_stage3 / renovate_stage3)
* ``outputs/checkpoints/stage2/<name>/meta.json`` and ``data/results_snapshot/stage2/checkpoints_r5/*``  (train_stage2 curves)
* ``data/results_snapshot/**/*.md|csv``                                                                  (paper tables)

Nothing here is computed from scratch — the registry only reads what the scripts wrote, so numbers shown in the UI are
exactly the numbers in the files (no re-aggregation that could silently disagree with the thesis tables).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

STAGE1_KEYS = ["ndcg@5", "ndcg@10", "map", "mrr", "spearman", "kendall_tau", "pairwise_acc", "hit@5"]
STAGE2_KEYS = ["overall_pass", "node_deviation_pct", "density_deviation_pct", "aspl_deviation_pct", "target_edge_recall_pct", "target_edge_precision_pct", "n_components", "inference_time_s"]
STAGE3_KEYS = ["planar_rate", "chamfer_ratio", "n_entrances", "n_vertical_cores", "n_atria", "corridor_ratio", "sharp_angle_rate"]


@dataclass
class ExperimentRecord:
    name: str
    family: str  # stage1 | stage1_ablation | stage2_train | stage2_eval | stage3_eval | renovation | table
    path: Path
    split: str = ""
    seeds: list[int] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    modified: float = 0.0

    def row(self) -> dict[str, Any]:
        r = {"name": self.name, "family": self.family, "split": self.split, "seeds": len(self.seeds) or "", "path": str(self.path.relative_to(ROOT)) if self.path.is_relative_to(ROOT) else str(self.path), "modified": pd.Timestamp(self.modified, unit="s").strftime("%Y-%m-%d %H:%M") if self.modified else ""}
        r.update(self.metrics)
        return r


def _read_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _num(d: dict[str, Any], keys: list[str]) -> dict[str, float]:
    out = {}
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and v == v:
            out[k] = float(v)
    return out


class ExperimentRegistry:
    def __init__(self, roots: list[Path] | None = None, snapshot: Path | None = None):
        self.roots = roots or [ROOT / "outputs" / "experiments"]
        self.snapshot = snapshot or ROOT / "data" / "results_snapshot"
        self.records: list[ExperimentRecord] = []

    # ------------------------------------------------------------------ scanning
    def scan(self) -> "ExperimentRegistry":
        recs: list[ExperimentRecord] = []
        for root in self.roots:
            if not root.exists():
                continue
            for d in sorted(p for p in root.rglob("*") if p.is_dir()):
                recs.extend(self._classify(d))
        recs.extend(self._checkpoints())
        seen = set()
        self.records = []
        for r in recs:
            key = (r.family, str(r.path), r.split)
            if key not in seen:
                seen.add(key)
                self.records.append(r)
        self.records.sort(key=lambda r: -r.modified)
        return self

    def _classify(self, d: Path) -> list[ExperimentRecord]:
        out: list[ExperimentRecord] = []
        tm = d / "test_metrics.json"
        if tm.exists():  # train_stage1 result dir
            for split in ("test", "val"):
                f = d / f"{split}_metrics.json"
                if f.exists():
                    m = _read_json(f)
                    if m.get("n_queries", 0):
                        meta = _read_json(d / "checkpoint" / "checkpoint_meta.json")
                        out.append(ExperimentRecord(d.name, "stage1", d, split, [int(meta.get("config", {}).get("seed", 0))] if meta else [], _num(m, STAGE1_KEYS), {"ranker": meta.get("config", {}).get("stage1", {}).get("ranker", {}).get("name", ""), "n_queries": m.get("n_queries")}, f.stat().st_mtime))
        seeds = sorted(d.glob("seed_*/run.json"))
        if seeds:  # ablation variant dir
            for split in ("test", "val"):
                rows = []
                for rj in seeds:
                    run = _read_json(rj)
                    m = (run.get("metrics") or {}).get(split) or run.get(f"{split}_metrics") or {}
                    if not m:
                        mf = rj.parent / f"{split}_metrics.json"
                        m = _read_json(mf) if mf.exists() else {}
                    if m:
                        rows.append({**_num(m, STAGE1_KEYS), "seed": run.get("seed")})
                if rows:
                    df = pd.DataFrame(rows)
                    mean = {k: float(df[k].mean()) for k in STAGE1_KEYS if k in df}
                    std = {f"{k}_std": float(df[k].std(ddof=0)) for k in STAGE1_KEYS if k in df}
                    r0 = _read_json(seeds[0])
                    out.append(ExperimentRecord(f"{d.parent.name}/{d.name}", "stage1_ablation", d, split, [int(s) for s in df["seed"].dropna()], {**mean, **std}, {"ranker": r0.get("ranker", ""), "variant": r0.get("variant", ""), "experiment_name": r0.get("experiment_name", "")}, max(s.stat().st_mtime for s in seeds)))
        agg = d / "aggregate.json"
        if agg.exists():
            a = _read_json(agg)
            if "overall_pass" in a:
                out.append(ExperimentRecord(d.name, "stage2_eval", d, a.get("split", "test"), [], _num(a, STAGE2_KEYS), {"generator": a.get("generator"), "n_samples": a.get("n_samples")}, agg.stat().st_mtime))
        summ = d / "summary.json"
        if summ.exists():
            s = _read_json(summ)
            if "num_cycles" in s or "avg_shortest_path" in s:  # renovation summary
                metrics = {}
                for k, v in s.items():
                    if isinstance(v, dict) and "improved_rate" in v:
                        if v.get("improved_rate") is not None:
                            metrics[f"{k}_improved"] = float(v["improved_rate"])
                        if v.get("delta_mean") is not None:
                            metrics[f"{k}_delta"] = float(v["delta_mean"])
                out.append(ExperimentRecord(d.name, "renovation", d, "", [], metrics, {"n_floors": s.get("n_floors"), "generator": s.get("generator"), "status_counts": s.get("status_counts")}, summ.stat().st_mtime))
            elif "self" in s:  # stage-3 evaluation summary
                m = (s.get("self") or {}).get("mean") or {}
                metrics = _num(m, STAGE3_KEYS + ["crossings", "chamfer_m", "chamfer_rand_m", "n_nodes"])
                if "planar_rate" not in metrics and "crossings" in m:
                    pr = (s.get("self") or {}).get("planar_rate")
                    if isinstance(pr, (int, float)):
                        metrics["planar_rate"] = float(pr)
                if "chamfer_ratio" not in metrics and m.get("chamfer_rand_m"):
                    metrics["chamfer_ratio"] = float(m["chamfer_m"]) / float(m["chamfer_rand_m"])
                out.append(ExperimentRecord(d.name, "stage3_eval", d, s.get("split", ""), [], metrics, {"n_samples": s.get("n_samples"), "status_counts": s.get("status_counts")}, summ.stat().st_mtime))
        return out

    def _checkpoints(self) -> list[ExperimentRecord]:
        out = []
        for root in (ROOT / "outputs" / "checkpoints" / "stage2", self.snapshot / "stage2" / "checkpoints_r5", self.snapshot / "stage2" / "checkpoints"):
            if not root.exists():
                continue
            for meta in sorted(root.glob("*/meta.json")):
                m = _read_json(meta)
                h = m.get("history") or {}
                metrics = {}
                for k, v in h.items():
                    if isinstance(v, list) and v and isinstance(v[-1], (int, float)):
                        metrics[f"final_{k}"] = float(v[-1])
                        metrics[f"best_{k}"] = float(max(v) if "acc" in k else min(v))
                metrics["epochs"] = float(len(next(iter(h.values()), [])))
                out.append(ExperimentRecord(f"{root.parent.name}/{meta.parent.name}" if root.parent.name != "checkpoints" else meta.parent.name, "stage2_train", meta.parent, "", [], metrics, {"hp": m.get("hp"), "feature_version": m.get("feature_version"), "n_states": m.get("n_states")}, meta.stat().st_mtime))
        return out

    # ------------------------------------------------------------------ queries
    def table(self, family: str | None = None, split: str | None = None) -> pd.DataFrame:
        rows = [r.row() for r in self.records if (family is None or r.family == family) and (split is None or not r.split or r.split == split)]
        if not rows:
            return pd.DataFrame(columns=["name", "family", "split"])
        df = pd.DataFrame(rows)
        lead = [c for c in ("name", "family", "split", "seeds", "modified") if c in df]
        return df[lead + [c for c in df.columns if c not in lead and c != "path"] + ["path"]]

    def families(self) -> list[str]:
        return sorted({r.family for r in self.records})

    def get(self, name: str, family: str | None = None) -> ExperimentRecord | None:
        for r in self.records:
            if r.name == name and (family is None or r.family == family):
                return r
        return None

    def compare(self, names: list[str], split: str = "test") -> pd.DataFrame:
        recs = [r for r in self.records if r.name in names and (not r.split or r.split == split)]
        if not recs:
            return pd.DataFrame()
        df = pd.DataFrame([r.row() for r in recs]).set_index("name")
        num = df.select_dtypes("number").T
        num.index.name = "metric"
        return num

    def curves(self, name: str) -> dict[str, list[float]]:
        """Training curves: Stage-2 checkpoint meta.json history, or the first-seed history of a Stage-1 ablation run."""
        r = self.get(name, "stage2_train")
        if r is not None:
            return {k: v for k, v in (_read_json(r.path / "meta.json").get("history") or {}).items() if isinstance(v, list)}
        r = self.get(name, "stage1_ablation")
        if r is not None:
            seeds = sorted(r.path.glob("seed_*/run.json"))
            if seeds:
                return {k: v for k, v in (_read_json(seeds[0]).get("history") or {}).items() if isinstance(v, list)}
        r = self.get(name, "stage1")
        if r is not None:
            meta = _read_json(r.path / "checkpoint" / "checkpoint_meta.json")
            return {k: v for k, v in (meta.get("history") or {}).items() if isinstance(v, list)}
        return {}

    def per_query(self, name: str, split: str = "test") -> pd.DataFrame | None:
        r = self.get(name)
        if r is None:
            return None
        for cand in (r.path / f"{split}_per_query.csv", r.path / "per_sample.csv", r.path / "per_floor.csv"):
            if cand.exists():
                return pd.read_csv(cand)
        seeds = sorted(r.path.glob(f"seed_*/{split}_per_query.csv"))
        if seeds:
            return pd.concat([pd.read_csv(s).assign(seed=s.parent.name) for s in seeds], ignore_index=True)
        return None

    def checkpoints(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            if r.family == "stage2_train":
                pt = list(r.path.glob("*.pt"))
                rows.append({"name": r.name, "stage": 2, "path": str(r.path), "file": pt[0].name if pt else "", "size_MB": round(pt[0].stat().st_size / 1e6, 2) if pt else None, "epochs": int(r.metrics.get("epochs", 0)), **{k: v for k, v in r.metrics.items() if k.startswith("final_")}, "hp": json.dumps(r.meta.get("hp"), ensure_ascii=False)})
            elif r.family == "stage1" and r.split == "test":
                ck = r.path / "checkpoint"
                jl = list(ck.glob("*.joblib")) if ck.exists() else []
                rows.append({"name": r.name, "stage": 1, "path": str(ck), "file": jl[0].name if jl else "", "size_MB": round(jl[0].stat().st_size / 1e6, 2) if jl else None, "ranker": r.meta.get("ranker", "")})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ paper tables
    def paper_tables(self) -> list[Path]:
        if not self.snapshot.exists():
            return []
        return sorted([p for p in self.snapshot.rglob("*.md")] + [p for p in self.snapshot.rglob("*.csv") if "per_seed" not in p.name])

    # ------------------------------------------------------------------ export
    @staticmethod
    def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
        if df.empty:
            return "(empty)"
        d = df.copy()
        for c in d.select_dtypes("number").columns:
            d[c] = d[c].map(lambda v: f"{v:.{digits}f}" if pd.notna(v) else "")
        cols = list(d.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
        for _, row in d.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    @staticmethod
    def to_latex(df: pd.DataFrame, digits: int = 3, caption: str = "", label: str = "") -> str:
        if df.empty:
            return "% empty table"
        d = df.copy()
        for c in d.select_dtypes("number").columns:
            d[c] = d[c].map(lambda v: f"{v:.{digits}f}" if pd.notna(v) else "")
        body = " \\\\\n".join(" & ".join(str(x).replace("_", "\\_").replace("%", "\\%") for x in row) for row in d.itertuples(index=False))
        head = " & ".join(str(c).replace("_", "\\_") for c in d.columns)
        return f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{'l' * len(d.columns)}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"
