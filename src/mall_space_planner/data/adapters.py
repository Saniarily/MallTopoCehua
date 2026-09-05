"""Dataset adapters registered in the component registry.

An adapter turns a *source* (legacy CSV+graphs, or the synthetic generator) into a
:class:`~mall_space_planner.data.case_db.CaseDatabase` with graph-derived metric columns
and a leakage-safe split. Selected via ``configs/data/*.yaml``::

    dataset:
      adapter: legacy          # or: synthetic
      params: {...}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.data.legacy_adapter import (
    LegacyDataSpec,
    iter_graphs,
    load_main_table,
    split_floor_id,
)
from mall_space_planner.data.splits import grouped_split
from mall_space_planner.data.synthetic import generate_synthetic_dataset
from mall_space_planner.registry import register
from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.topology.metrics import compute_topology_metrics
from mall_space_planner.utils.logging import get_logger
from mall_space_planner.utils.repro import collect_environment_info

logger = get_logger(__name__)

GRAPH_METRIC_COLS: tuple[str, ...] = (
    "g_num_nodes",
    "g_num_edges",
    "g_density",
    "g_avg_degree",
    "g_diameter",
    "g_avg_shortest_path",
    "g_num_cycles",
    "g_n_components",
    "g_clustering",
    "g_degree_entropy",
    "g_max_betweenness",
)


def graph_metric_row(g: TopologyGraph | None) -> dict[str, float | None]:
    if g is None:
        return {c: None for c in GRAPH_METRIC_COLS}
    m = compute_topology_metrics(g)
    return {
        "g_num_nodes": m.num_nodes,
        "g_num_edges": m.num_edges,
        "g_density": m.density,
        "g_avg_degree": m.avg_degree,
        "g_diameter": m.diameter,
        "g_avg_shortest_path": m.avg_shortest_path,
        "g_num_cycles": m.num_cycles,
        "g_n_components": m.n_components,
        "g_clustering": m.clustering,
        "g_degree_entropy": m.degree_entropy,
        "g_max_betweenness": m.max_betweenness,
    }


class BaseDatasetAdapter(ABC):
    """Interface: ``build()`` returns a :class:`CaseDatabase`."""

    @abstractmethod
    def build(self) -> CaseDatabase: ...


def _build_case_db(
    df: pd.DataFrame,
    spec: LegacyDataSpec,
    split_cfg: dict[str, Any],
    source_name: str,
    normalize_positions: bool = False,
    max_rows: int | None = None,
) -> CaseDatabase:
    df = df.copy()
    if max_rows:
        df = df.head(max_rows)
    df["floor_index"] = [split_floor_id(f)[1] for f in df[spec.id_col]]

    graphs: dict[str, TopologyGraph] = {}
    metric_rows: list[dict] = []
    for fid, g in tqdm(iter_graphs(spec, df[spec.id_col].tolist(), normalize_positions), total=len(df), desc="graphs", leave=False):
        if g is not None:
            graphs[fid] = g
        metric_rows.append(graph_metric_row(g))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(metric_rows)], axis=1)
    df["has_graph"] = df[spec.id_col].isin(graphs.keys())
    n_missing = int((~df["has_graph"]).sum())
    if n_missing:
        logger.warning("%d/%d floors have no graph files", n_missing, len(df))

    res = grouped_split(
        df,
        group_col=spec.mall_id_col,
        test_ratio=float(split_cfg.get("test_ratio", 0.1)),
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        seed=int(split_cfg.get("seed", 42)),
        stratify_col=split_cfg.get("stratify_col"),
    )
    df["split"] = "train"
    df.loc[df[spec.id_col].isin(res.val[spec.id_col]), "split"] = "val"
    df.loc[df[spec.id_col].isin(res.test[spec.id_col]), "split"] = "test"

    manifest = {
        "source": source_name,
        "main_table_csv": str(spec.main_table_csv),
        "graph_dir": str(spec.graph_dir),
        "id_col": spec.id_col,
        "mall_id_col": spec.mall_id_col,
        "label_col": spec.label_col,
        "city_cluster_col": spec.city_cluster_col,
        "total_area_col": spec.total_area_col,
        "query_cols": spec.query_cols,
        "metric_cols": spec.metric_cols,
        "graph_metric_cols": list(GRAPH_METRIC_COLS),
        "split": {**split_cfg, "group_col": spec.mall_id_col, **res.summary()},
        "n_missing_graphs": n_missing,
        "environment": collect_environment_info(),
    }
    return CaseDatabase(cases=df, graphs=graphs, manifest=manifest)


@register("dataset_adapter", "legacy")
class LegacyDatasetAdapter(BaseDatasetAdapter):
    """Adapter for the real legacy data (paths given in ``configs/data/legacy.yaml``)."""

    def __init__(
        self,
        main_table_csv: str,
        graph_dir: str,
        split: dict[str, Any] | None = None,
        graph_suffix_edge: str = "_M_simplified.csv",
        graph_suffix_node: str = "_M_simplified_node_attributes.csv",
        features: dict[str, Any] | None = None,
        normalize_positions: bool = False,
        max_rows: int | None = None,
    ) -> None:
        f = features or {}
        self.spec = LegacyDataSpec(
            main_table_csv=Path(main_table_csv).expanduser(),
            graph_dir=Path(graph_dir).expanduser(),
            graph_suffix_edge=graph_suffix_edge,
            graph_suffix_node=graph_suffix_node,
            **{k: v for k, v in f.items() if k in LegacyDataSpec.__dataclass_fields__},
        )
        self.split_cfg = split or {}
        self.normalize_positions = normalize_positions
        self.max_rows = max_rows

    def build(self) -> CaseDatabase:
        if not self.spec.available():
            raise FileNotFoundError(
                f"Legacy data not found: {self.spec.main_table_csv} / {self.spec.graph_dir}. "
                "Set the paths in configs/data/legacy.yaml or use the synthetic adapter."
            )
        df = load_main_table(self.spec)
        return _build_case_db(df, self.spec, self.split_cfg, "legacy", self.normalize_positions, self.max_rows)


@register("dataset_adapter", "synthetic")
class SyntheticDatasetAdapter(BaseDatasetAdapter):
    """Generates synthetic legacy-format data on the fly (for smoke tests / demos)."""

    def __init__(
        self,
        out_dir: str,
        n_malls: int = 40,
        max_floors: int = 4,
        n_stage2: int = 120,
        seed: int = 0,
        split: dict[str, Any] | None = None,
        regenerate: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.n_malls, self.max_floors, self.n_stage2, self.seed = n_malls, max_floors, n_stage2, seed
        self.split_cfg = split or {}
        self.regenerate = regenerate

    def build(self) -> CaseDatabase:
        main = self.out_dir / "main_table.csv"
        if self.regenerate or not main.exists():
            generate_synthetic_dataset(self.out_dir, self.n_malls, self.max_floors, self.n_stage2, self.seed)
        spec = LegacyDataSpec(main_table_csv=main, graph_dir=self.out_dir / "graphs")
        df = load_main_table(spec)
        db = _build_case_db(df, spec, self.split_cfg, "synthetic")
        db.manifest["synthetic"] = True
        return db
