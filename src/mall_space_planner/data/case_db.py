"""Processed case database: the single artefact consumed by Stage-1 pipelines.

Layout on disk (``data/processed/<name>/``)::

    cases.parquet|csv     one row per floor: ids, condition features, metric features, label, split
    graphs.jsonl          one JSON per line: {"floor_id": ..., "adjacency": {...}, "positions": {...}}
    manifest.json         provenance: source spec, column lists, counts, split protocol, env info

The database is deliberately simple (pandas + JSONL) so that it is inspectable and
portable; no runtime dependency on the legacy repository or on PyTorch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mall_space_planner.schemas import (
    LayoutType,
    MallCase,
    PlanningCondition,
    TopologyGraph,
    TopologyMetrics,
    TopologyPrototype,
)
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CaseDatabase:
    """In-memory case database with lazy graph access."""

    cases: pd.DataFrame
    graphs: dict[str, TopologyGraph]
    manifest: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ properties
    @property
    def id_col(self) -> str:
        return self.manifest.get("id_col", "floor_id")

    @property
    def mall_id_col(self) -> str:
        return self.manifest.get("mall_id_col", "mall_id")

    @property
    def label_col(self) -> str:
        return self.manifest.get("label_col", "total_score")

    @property
    def query_cols(self) -> list[str]:
        return list(self.manifest.get("query_cols", []))

    @property
    def metric_cols(self) -> list[str]:
        return list(self.manifest.get("metric_cols", []))

    @property
    def graph_metric_cols(self) -> list[str]:
        return list(self.manifest.get("graph_metric_cols", []))

    def split(self, name: str) -> pd.DataFrame:
        if "split" not in self.cases.columns:
            raise KeyError("case table has no 'split' column")
        return self.cases[self.cases["split"] == name].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.cases)

    # ------------------------------------------------------------------ objects
    def get_graph(self, floor_id: str) -> TopologyGraph | None:
        return self.graphs.get(str(floor_id))

    def get_case(self, floor_id: str) -> MallCase:
        row = self.cases.loc[self.cases[self.id_col] == str(floor_id)]
        if row.empty:
            raise KeyError(floor_id)
        r = row.iloc[0].to_dict()
        cond = PlanningCondition(**{k: (None if pd.isna(v) else v) for k, v in r.items() if k in PlanningCondition.model_fields})
        g = self.get_graph(floor_id)
        lt = r.get("layout_type")
        proto = None
        if g is not None:
            metrics = TopologyMetrics(**{k: (None if pd.isna(r.get(k)) else r.get(k)) for k in self.metric_cols + self.graph_metric_cols if k in r})
            proto = TopologyPrototype(
                prototype_id=str(floor_id),
                graph=g,
                layout_type=LayoutType(lt) if isinstance(lt, str) and lt in {t.value for t in LayoutType} else None,
                metrics=metrics,
                source_case_id=str(r.get(self.mall_id_col)),
                quality_score=None if pd.isna(r.get(self.label_col)) else float(r[self.label_col]),
            )
        return MallCase(
            floor_id=str(floor_id),
            mall_id=str(r.get(self.mall_id_col)),
            floor_index=None if pd.isna(r.get("floor_index")) else int(r["floor_index"]),
            condition=cond,
            prototype=proto,
            total_score=None if pd.isna(r.get(self.label_col)) else float(r[self.label_col]),
        )

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.cases.to_parquet(out_dir / "cases.parquet", index=False)
            table_file = "cases.parquet"
        except (ImportError, ValueError):
            self.cases.to_csv(out_dir / "cases.csv", index=False)
            table_file = "cases.csv"
        with open(out_dir / "graphs.jsonl", "w", encoding="utf-8") as f:
            for fid, g in self.graphs.items():
                f.write(json.dumps({"floor_id": fid, "adjacency": g.adjacency, "positions": g.positions, "node_attrs": g.node_attrs}, ensure_ascii=False) + "\n")
        manifest = dict(self.manifest)
        manifest["table_file"] = table_file
        manifest["n_cases"] = int(len(self.cases))
        manifest["n_graphs"] = int(len(self.graphs))
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        logger.info("CaseDatabase saved to %s (%d cases, %d graphs)", out_dir, len(self.cases), len(self.graphs))
        return out_dir

    @classmethod
    def load(cls, in_dir: str | Path) -> CaseDatabase:
        in_dir = Path(in_dir)
        with open(in_dir / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        table_file = manifest.get("table_file", "cases.parquet")
        table_path = in_dir / table_file
        cases = pd.read_parquet(table_path) if table_path.suffix == ".parquet" else pd.read_csv(table_path)
        cases[manifest.get("id_col", "floor_id")] = cases[manifest.get("id_col", "floor_id")].astype(str)
        graphs: dict[str, TopologyGraph] = {}
        gpath = in_dir / "graphs.jsonl"
        if gpath.exists():
            with open(gpath, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    graphs[str(rec["floor_id"])] = TopologyGraph(
                        adjacency=rec["adjacency"],
                        positions={k: tuple(v) for k, v in rec.get("positions", {}).items()},
                        node_attrs=rec.get("node_attrs", {}),
                    )
        return cls(cases=cases, graphs=graphs, manifest=manifest)
