"""Read-only adapter for the legacy *MallTopoRanker* data.

Provenance (facts confirmed by reading the legacy repository, not assumptions):

* Main table: ``DATASET_3+6类城市_with_rkn_ratios_and_topo_metrics.csv`` with at least the
  columns ``floor_id, mall_id, city_cluster, total_score`` + the 10 query columns and 4
  metric columns listed in ``config.yaml`` (``features.query_cols`` / ``metric_cols``).
* Graph files per floor in ``graph_dir``:
  ``{floor_id}_M_simplified.csv`` (edge list, columns ``Source, Target``; separator
  auto-detected) and ``{floor_id}_M_simplified_node_attributes.csv`` (columns
  ``Node_ID, Total_L_Neighbors`` and optional ``CenterPoint`` formatted ``"(x, y)"``).
* ``floor_id`` has the form ``{mall_id}_{floor_index}`` (e.g. ``B000A08791_3``).
* Node ids are ``M###`` corridor-junction nodes only.

This module re-implements the tiny parsing logic of ``legacy/src/utils_graph_io.py``
(≈40 lines) as independent, tested functions; nothing is imported from the old repo.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mall_space_planner.schemas import TopologyGraph
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)

_CENTER_RE = re.compile(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)")

DEFAULT_QUERY_COLS: tuple[str, ...] = (
    "people",
    "GDP_2023",
    "PCDI_2023",
    "TP_2023",
    "mall_area_count",
    "nearest_distance_km",
    "count_1km",
    "count_2km",
    "total_area",
    "Tx",
)
DEFAULT_METRIC_COLS: tuple[str, ...] = ("L1_density", "L2_diameter", "L2_complexity", "L2_integration")

# Additional floor-level descriptors confirmed in the real 97-column main table (2026-09-05 audit).
# They are candidate-side (prototype) features; all are optional and only used when present.
DEFAULT_EXTRA_METRIC_COLS: tuple[str, ...] = (
    "corridor_area_ratio",
    "area_corridor",
    "total_perimeter",
    "Topological_Diameter",
    "Topological_Complexity",
    "Betweenness_Variance",
    "Global_Integration",
    "L1_diameter",
    "L1_clustering",
    "L2_bc_variance",
    "L3_efficiency",
    "L3_morphology",
    "L3_balance",
    "L3_integration",
    "s_mean_degree",
    "s_mean_closeness_centrality",
    "s_mean_betweenness_centrality",
    "s_mean_mean_depth",
)
# Categorical / passthrough columns kept for filtering, display and explanation.
DEFAULT_PASSTHROUGH_COLS: tuple[str, ...] = (
    "floor",
    "type8",
    "cityname",
    "adname",
    "lon",
    "lat",
    "mall_category_simplified",
    "city_cluster_6",
    "score_num",
    "score_facility",
    "score_environment",
    "score_service",
)


@dataclass
class LegacyDataSpec:
    """Locations and column names of the legacy dataset (mirrors legacy ``config.yaml``)."""

    main_table_csv: Path
    graph_dir: Path
    graph_suffix_edge: str = "_M_simplified.csv"
    graph_suffix_node: str = "_M_simplified_node_attributes.csv"
    id_col: str = "floor_id"
    mall_id_col: str = "mall_id"
    label_col: str = "total_score"
    city_cluster_col: str = "city_cluster"
    total_area_col: str = "total_area"
    query_cols: list[str] = field(default_factory=lambda: list(DEFAULT_QUERY_COLS))
    metric_cols: list[str] = field(default_factory=lambda: list(DEFAULT_METRIC_COLS))
    extra_metric_cols: list[str] = field(default_factory=lambda: list(DEFAULT_EXTRA_METRIC_COLS))
    passthrough_cols: list[str] = field(default_factory=lambda: list(DEFAULT_PASSTHROUGH_COLS))
    layout_col: str = "type8"

    @classmethod
    def from_config(cls, cfg: dict) -> LegacyDataSpec:
        d, f = cfg.get("data", {}), cfg.get("features", {})
        return cls(
            main_table_csv=Path(d["main_table_csv"]).expanduser(),
            graph_dir=Path(d["graph_dir"]).expanduser(),
            graph_suffix_edge=d.get("graph_suffix_edge", "_M_simplified.csv"),
            graph_suffix_node=d.get("graph_suffix_node", "_M_simplified_node_attributes.csv"),
            id_col=f.get("id_col", "floor_id"),
            mall_id_col=f.get("mall_id_col", "mall_id"),
            label_col=f.get("label_col", "total_score"),
            city_cluster_col=f.get("city_cluster_col", "city_cluster"),
            total_area_col=f.get("total_area_col", "total_area"),
            query_cols=list(f.get("query_cols", DEFAULT_QUERY_COLS)),
            metric_cols=list(f.get("metric_cols", DEFAULT_METRIC_COLS)),
            extra_metric_cols=list(f.get("extra_metric_cols", DEFAULT_EXTRA_METRIC_COLS)),
            passthrough_cols=list(f.get("passthrough_cols", DEFAULT_PASSTHROUGH_COLS)),
            layout_col=f.get("layout_col", "type8"),
        )

    def available(self) -> bool:
        return self.main_table_csv.exists() and self.graph_dir.exists()


# --------------------------------------------------------------------------- parsing
def parse_centerpoint(value: object) -> tuple[float, float] | None:
    """Parse ``"(123, 456)"`` → ``(123.0, 456.0)``; return ``None`` for missing/malformed."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = _CENTER_RE.search(str(value))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def split_floor_id(floor_id: str) -> tuple[str, int | None]:
    """``"B000A08791_3"`` → ``("B000A08791", 3)``. Floor index ``None`` if absent."""
    fid = str(floor_id)
    if "_" not in fid:
        return fid, None
    mall, idx = fid.rsplit("_", 1)
    try:
        return mall, int(idx)
    except ValueError:
        return fid, None


def load_graph_csv(edge_csv: Path, node_csv: Path, normalize_positions: bool = False) -> TopologyGraph:
    """Load one legacy graph (edge list + node attributes) into a :class:`TopologyGraph`.

    Args:
        edge_csv: ``*_M_simplified.csv`` with columns ``Source, Target``.
        node_csv: ``*_M_simplified_node_attributes.csv`` with ``Node_ID, Total_L_Neighbors[, CenterPoint]``.
        normalize_positions: If True, centre coordinates and scale to ``[-1, 1]`` per graph
            (this is what the legacy PyG cache did). Raw pixel coordinates are kept otherwise.
    """
    edf = pd.read_csv(edge_csv, sep=None, engine="python")
    if not {"Source", "Target"}.issubset(edf.columns):
        raise ValueError(f"{edge_csv}: expected columns Source/Target, got {list(edf.columns)}")
    ndf = pd.read_csv(node_csv, sep=None, engine="python")
    if not {"Node_ID", "Total_L_Neighbors"}.issubset(ndf.columns):
        raise ValueError(f"{node_csv}: expected Node_ID/Total_L_Neighbors, got {list(ndf.columns)}")

    node_ids = [str(x) for x in ndf["Node_ID"].tolist()]
    known = set(node_ids)
    adjacency: dict[str, list[str]] = {n: [] for n in node_ids}
    dropped = 0
    for s, t in zip(edf["Source"].astype(str), edf["Target"].astype(str), strict=True):
        if s in known and t in known:
            adjacency[s].append(t)
        else:
            dropped += 1
    if dropped:
        logger.debug("%s: dropped %d edges referencing unknown nodes", edge_csv.name, dropped)

    positions: dict[str, tuple[float, float]] = {}
    if "CenterPoint" in ndf.columns:
        for nid, cp in zip(node_ids, ndf["CenterPoint"].tolist(), strict=True):
            p = parse_centerpoint(cp)
            if p is not None:
                positions[nid] = p
        if normalize_positions and positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            sx = max(1.0, max(abs(x - cx) for x in xs))
            sy = max(1.0, max(abs(y - cy) for y in ys))
            positions = {k: ((x - cx) / sx, (y - cy) / sy) for k, (x, y) in positions.items()}

    node_attrs = {
        nid: {"total_l_neighbors": float(v) if pd.notna(v) else 0.0}
        for nid, v in zip(node_ids, ndf["Total_L_Neighbors"].tolist(), strict=True)
    }
    node_types = {nid: nid[0] if nid and nid[0].isalpha() else "M" for nid in node_ids}
    return TopologyGraph(adjacency=adjacency, positions=positions, node_attrs=node_attrs, node_types=node_types)


# --------------------------------------------------------------------------- table
# Spreadsheet error tokens observed in the real export (treated as missing, counted in the audit).
NON_NUMERIC_TOKENS: tuple[str, ...] = ("#DIV/0!", "#N/A", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#NULL!", "NA", "N/A", "nan", "None", "")


def coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Force ``cols`` to float, turning spreadsheet error strings into NaN.

    Returns the frame and a per-column count of values that were non-numeric (excluding
    genuine NaN), so that data quality issues are reported rather than silently hidden.
    """
    report: dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            continue
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            continue
        as_str = col.astype("string").str.strip()
        was_present = col.notna() & ~as_str.isin(NON_NUMERIC_TOKENS[-4:])  # ignore blank-ish tokens
        num = pd.to_numeric(as_str.replace(list(NON_NUMERIC_TOKENS), pd.NA), errors="coerce")
        bad = int((was_present & num.isna()).sum())
        if bad:
            report[c] = bad
        df[c] = num.astype(float)
    return df, report


def load_main_table(spec: LegacyDataSpec) -> pd.DataFrame:
    """Load the main CSV, validate configured columns and coerce numeric columns.

    Non-numeric tokens such as ``#DIV/0!`` become NaN; their counts are attached to the
    frame as ``df.attrs["coercion_report"]`` for the audit.
    """
    df = pd.read_csv(spec.main_table_csv, low_memory=False)
    required = [spec.id_col, spec.mall_id_col, spec.label_col, spec.city_cluster_col, *spec.query_cols, *spec.metric_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Main table is missing configured columns: {missing}")
    df[spec.id_col] = df[spec.id_col].astype(str)
    df[spec.mall_id_col] = df[spec.mall_id_col].astype(str)
    n_dup = int(df[spec.id_col].duplicated().sum())
    if n_dup:
        logger.warning("Dropping %d duplicated %s rows (keeping first occurrence)", n_dup, spec.id_col)
        df = df.drop_duplicates(subset=[spec.id_col], keep="first").reset_index(drop=True)
    # Standardised layout column (real table: `type8`; synthetic: `layout_type`).
    if "layout_type" not in df.columns and spec.layout_col in df.columns:
        df["layout_type"] = df[spec.layout_col].astype(str).where(df[spec.layout_col].notna(), None)
    numeric_cols = [spec.label_col, spec.total_area_col, *spec.query_cols, *spec.metric_cols, *spec.extra_metric_cols]
    df, report = coerce_numeric_columns(df, numeric_cols)
    if report:
        logger.warning("Non-numeric tokens coerced to NaN: %s", report)
    df.attrs["coercion_report"] = report
    return df


def graph_paths_for(spec: LegacyDataSpec, floor_id: str) -> tuple[Path, Path]:
    return (
        spec.graph_dir / f"{floor_id}{spec.graph_suffix_edge}",
        spec.graph_dir / f"{floor_id}{spec.graph_suffix_node}",
    )


def iter_graphs(spec: LegacyDataSpec, floor_ids: Iterable[str], normalize_positions: bool = False):
    """Yield ``(floor_id, TopologyGraph | None)`` for each id; ``None`` when files are missing."""
    for fid in floor_ids:
        e, n = graph_paths_for(spec, fid)
        if e.exists() and n.exists():
            try:
                yield fid, load_graph_csv(e, n, normalize_positions=normalize_positions)
            except Exception as exc:  # noqa: BLE001 - report, do not crash a bulk conversion
                logger.warning("Failed to load graph %s: %s", fid, exc)
                yield fid, None
        else:
            yield fid, None
