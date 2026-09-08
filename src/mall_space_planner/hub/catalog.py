"""Case catalogue for the data browser: statistics, filters, anomaly report, topology / outline access.

Pure data layer – returns DataFrames / dicts / shapely objects; the UI draws them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.utils.config import resolve_config

ROOT = Path(__file__).resolve().parents[3]


def locate_database(preferred: str | None = None) -> tuple[Path, bool]:
    """Real processed DB if present, else the synthetic one. Returns (path, is_real)."""
    cands: list[Path] = []
    if preferred:
        cands.append(Path(preferred))
    try:
        s1 = resolve_config("configs/stage1/extra_trees.yaml", [])
        cands.append(Path(s1["data"]["processed_dir"]))
    except Exception:  # noqa: BLE001
        pass
    cands.append(ROOT / "data/processed/legacy")
    for c in cands:
        if (c / "manifest.json").exists():
            return c, "synthetic" not in c.name
    return ROOT / "data/processed/synthetic", False


@dataclass
class Catalog:
    db: CaseDatabase
    path: Path
    is_real: bool

    @classmethod
    def load(cls, preferred: str | None = None) -> Catalog:
        p, real = locate_database(preferred)
        return cls(db=CaseDatabase.load(str(p)), path=p, is_real=real)

    # ---- statistics ---------------------------------------------------------------------------------------------
    @property
    def cases(self) -> pd.DataFrame:
        return self.db.cases

    def overview(self) -> dict[str, Any]:
        df = self.cases
        d = {
            "n_floors": int(len(df)),
            "n_malls": int(df[self.db.mall_id_col].nunique()) if self.db.mall_id_col in df else None,
            "n_graphs": int(len(self.db.graphs)),
            "n_missing_graphs": int(self.db.manifest.get("n_missing_graphs", 0)),
            "splits": df["split"].value_counts().to_dict() if "split" in df else {},
            "layout_types": df["layout_type"].value_counts(dropna=False).to_dict() if "layout_type" in df else {},
            "city_clusters": df["city_cluster"].value_counts(dropna=False).to_dict() if "city_cluster" in df else {},
            "score": df[self.db.label_col].describe().to_dict() if self.db.label_col in df else {},
            "source": self.db.manifest.get("source"),
            "path": str(self.path),
            "is_real": self.is_real,
        }
        return d

    def numeric_columns(self) -> list[str]:
        return [c for c in self.cases.columns if pd.api.types.is_numeric_dtype(self.cases[c]) and c not in ("has_graph",)]

    def missing_report(self) -> pd.DataFrame:
        df = self.cases
        miss = df.isna().sum()
        out = pd.DataFrame({"column": miss.index, "n_missing": miss.values, "pct": (miss.values / max(len(df), 1) * 100).round(2)})
        return out[out["n_missing"] > 0].sort_values("n_missing", ascending=False).reset_index(drop=True)

    def anomalies(self) -> pd.DataFrame:
        """Data-quality warnings: constant columns, negative counts, score outside range, disconnected graphs,
        floors without graph, extreme z-scores (> 5 σ) on query columns."""
        df = self.cases
        rows: list[dict[str, Any]] = []
        for c in self.numeric_columns():
            s = df[c].dropna()
            if len(s) and s.nunique() == 1:
                rows.append({"type": "constant_column", "column": c, "n": int(len(s)), "detail": f"value = {s.iloc[0]}"})
            if c in self.db.query_cols and len(s) > 10:
                z = (s - s.mean()) / (s.std() + 1e-9)
                n = int((z.abs() > 5).sum())
                if n:
                    rows.append({"type": "extreme_values", "column": c, "n": n, "detail": "> 5 σ"})
            if ("count" in c or c in ("people", "total_area")) and (s < 0).any():
                rows.append({"type": "negative_values", "column": c, "n": int((s < 0).sum()), "detail": ""})
        lab = self.db.label_col
        if lab in df:
            # scale-aware: legacy scores live in [0, 5], synthetic scores in [0, 100]
            hi = 5.0 if float(df[lab].max()) <= 5.0 else 100.0
            bad = df[(df[lab] < 0) | (df[lab] > hi)]
            if len(bad):
                rows.append({"type": "score_out_of_range", "column": lab, "n": int(len(bad)), "detail": f"outside [0, {hi:g}]"})
            if self.db.mall_id_col in df:
                var = df.groupby(self.db.mall_id_col)[lab].nunique()
                rows.append({"type": "info", "column": lab, "n": int((var > 1).sum()), "detail": "malls whose floors have different scores (expected 0: mall-level score)"})
        if "g_n_components" in df:
            n = int((df["g_n_components"] > 1).sum())
            if n:
                rows.append({"type": "disconnected_graph", "column": "g_n_components", "n": n, "detail": "> 1 component"})
        if "has_graph" in df:
            n = int((~df["has_graph"].astype(bool)).sum())
            if n:
                rows.append({"type": "missing_graph", "column": "has_graph", "n": n, "detail": ""})
        cr = self.db.manifest.get("coercion_report")
        if isinstance(cr, dict):
            for c, v in cr.items():
                rows.append({"type": "coerced_values", "column": c, "n": int(v) if isinstance(v, (int, float)) else 0, "detail": "non-numeric cells (e.g. #DIV/0!) -> NaN"})
        return pd.DataFrame(rows, columns=["type", "column", "n", "detail"])

    # ---- filtering --------------------------------------------------------------------------------------------------
    def filter(self, layout_types: list[str] | None = None, clusters: list[int] | None = None, score_range: tuple[float, float] | None = None,
               area_range: tuple[float, float] | None = None, splits: list[str] | None = None, has_graph: bool | None = None, text: str | None = None) -> pd.DataFrame:
        df = self.cases
        m = pd.Series(True, index=df.index)
        if layout_types and "layout_type" in df:
            m &= df["layout_type"].isin(layout_types)
        if clusters and "city_cluster" in df:
            m &= df["city_cluster"].isin(clusters)
        if score_range and self.db.label_col in df:
            m &= df[self.db.label_col].between(*score_range)
        if area_range and "total_area" in df:
            m &= df["total_area"].between(*area_range)
        if splits and "split" in df:
            m &= df["split"].isin(splits)
        if has_graph is not None and "has_graph" in df:
            m &= df["has_graph"].astype(bool) == has_graph
        if text:
            m &= df[self.db.id_col].astype(str).str.contains(text, case=False) | df[self.db.mall_id_col].astype(str).str.contains(text, case=False)
        return df[m].reset_index(drop=True)

    # ---- one case ---------------------------------------------------------------------------------------------------
    def case(self, floor_id: str) -> dict[str, Any]:
        row = self.cases[self.cases[self.db.id_col].astype(str) == str(floor_id)]
        if row.empty:
            raise KeyError(floor_id)
        r = row.iloc[0].to_dict()
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}

    def topology(self, floor_id: str):  # noqa: ANN201
        return self.db.get_graph(str(floor_id))

    def topology_metrics(self, floor_id: str) -> dict[str, Any]:
        from mall_space_planner.topology.metrics import compute_topology_metrics

        g = self.topology(floor_id)
        return compute_topology_metrics(g).model_dump() if g is not None else {}

    def outline(self, floor_id: str):  # noqa: ANN201
        """Stage-3 outline of the floor if the mask / total.csv is reachable, else None."""
        try:
            from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths

            cfg = resolve_config("configs/data/legacy.yaml", [])
            ds = Stage3Dataset(Stage3Paths.from_config(cfg))
            if ds.paths.outline_mask(floor_id) or ds.paths.total_csv(floor_id):
                return ds.outline(floor_id, 0)
        except Exception:  # noqa: BLE001
            return None
        return None

    def real_network(self, floor_id: str):  # noqa: ANN201
        """(complete M network, skeleton, real positions, outline) when the graph CSVs are reachable, else None."""
        try:
            from mall_space_planner.data.corpus_builder import load_target_csv
            from mall_space_planner.data.legacy_adapter import load_graph_csv
            from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths
            from mall_space_planner.stage3.outline import m_positions_from_total_csv

            cfg = resolve_config("configs/data/legacy.yaml", [])
            ds = Stage3Dataset(Stage3Paths.from_config(cfg))
            gd = ds.paths.graph_dir
            for cand in [gd, ROOT / "tests/fixtures/graph_csv"]:
                if cand and (cand / f"{floor_id}_M.csv").exists() and (cand / f"{floor_id}_total.csv").exists():
                    full = load_target_csv(cand / f"{floor_id}_M.csv")
                    sk = load_graph_csv(cand / f"{floor_id}_M_simplified.csv", cand / f"{floor_id}_M_simplified_node_attributes.csv") if (cand / f"{floor_id}_M_simplified.csv").exists() else None
                    ds.paths.graph_dir = cand
                    outline = ds.outline(floor_id, 0)
                    gt = m_positions_from_total_csv(cand / f"{floor_id}_total.csv", outline)
                    return {"full": full, "skeleton": sk, "positions": gt, "outline": outline, "graph_dir": cand}
        except Exception:  # noqa: BLE001
            return None
        return None

    def compare(self, floor_ids: list[str]) -> pd.DataFrame:
        cols = [self.db.id_col, self.db.mall_id_col, "layout_type", "city_cluster", self.db.label_col, "total_area", *self.db.graph_metric_cols]
        cols = [c for c in cols if c in self.cases.columns]
        df = self.cases[self.cases[self.db.id_col].astype(str).isin([str(f) for f in floor_ids])][cols]
        return df.set_index(self.db.id_col).T

    def audit_markdown(self) -> str:
        p = self.path / "audit.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def manifest_json(self) -> str:
        return json.dumps(self.db.manifest, indent=1, ensure_ascii=False, default=str)


__all__ = ["Catalog", "locate_database"]
