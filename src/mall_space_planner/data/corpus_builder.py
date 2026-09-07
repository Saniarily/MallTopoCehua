"""Build the *real* Stage-2 corpus from per-floor CSV graph files.

For every floor ``{mall_id}_{k}`` the legacy export contains three files:

* ``{floor_id}_M_simplified.csv``                 – **skeleton** (core corridor prototype): ``Source,Target``
* ``{floor_id}_M_simplified_node_attributes.csv`` – skeleton node attributes: ``Node_ID,Total_L_Neighbors,CenterPoint``
* ``{floor_id}_M.csv``                            – **complete corridor key-point topology** (target):
  ``Source,Target,Shared_L_Count,Shared_L_Nodes``

Facts verified on the uploaded sample ``B000A0E928_1`` (see ``docs/data_audit.md`` §Stage-2 v2):
skeleton 25 nodes / 28 edges → target 50 nodes / 83 edges; **all** skeleton nodes and edges are
kept verbatim in the target (node subset ✓, edge recall 1.0); every new node attaches to 1–4
already-present nodes (never a chain that re-routes a skeleton edge); the target is planar and
connected. Hence the same *prototype-preserving expansion* formulation as before applies, but the
target graphs are dense corridor networks (mean degree ≈ 3.3) rather than tree-like LLM outputs, and
a new node typically closes a loop (≥ 2 anchors) instead of hanging off one anchor.

The builder writes one JSON-lines file with :class:`ExpansionSample` records enriched with
``mall_id`` (for leakage-free grouped splits) and ``split`` (train / val / test by mall, stratified
by city cluster – the *same* protocol as Stage 1). Conditions (city, layout type, area) are joined
from the main table when available; floors without a main-table row are kept with ``None``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from mall_space_planner.data.legacy_adapter import load_graph_csv, parse_centerpoint, split_floor_id
from mall_space_planner.data.sharegpt_adapter import ExpansionSample
from mall_space_planner.schemas import LayoutType, TopologyGraph
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)

SUFFIX_SKELETON_EDGE = "_M_simplified.csv"
SUFFIX_SKELETON_NODE = "_M_simplified_node_attributes.csv"
SUFFIX_TARGET_EDGE = "_M.csv"


@dataclass
class CorpusStats:
    n_floors_seen: int = 0
    n_written: int = 0
    n_missing_target: int = 0
    n_missing_skeleton: int = 0
    n_skeleton_not_subgraph: int = 0
    n_target_disconnected: int = 0
    n_no_growth: int = 0
    n_bad: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _read_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if not {"Source", "Target"}.issubset(df.columns):
        raise ValueError(f"{path}: expected Source/Target columns, got {list(df.columns)}")
    df["Source"] = df["Source"].astype(str).str.strip()
    df["Target"] = df["Target"].astype(str).str.strip()
    return df


def load_target_csv(path: Path) -> TopologyGraph:
    """``*_M.csv`` → TopologyGraph. ``Shared_L_Count`` is kept as edge attribute ``shared_l``."""
    df = _read_edges(path)
    adjacency: dict[str, list[str]] = {}
    edge_attrs: dict[str, dict] = {}
    for _, r in df.iterrows():
        s, t = r["Source"], r["Target"]
        if s == t:
            continue
        adjacency.setdefault(s, []).append(t)
        adjacency.setdefault(t, [])
        if "Shared_L_Count" in df.columns and pd.notna(r["Shared_L_Count"]):
            u, v = sorted((s, t))
            edge_attrs[f"{u}|{v}"] = {"shared_l": int(r["Shared_L_Count"])}
    node_types = {n: (n[0] if n and n[0].isalpha() else "M") for n in adjacency}
    return TopologyGraph(adjacency=adjacency, node_types=node_types, edge_attrs=edge_attrs)


def floor_ids_in(graph_dir: Path) -> list[str]:
    """All floor ids that have a complete-topology file (``*_M.csv``), sorted."""
    pat = re.compile(rf"^(.*){re.escape(SUFFIX_TARGET_EDGE)}$")
    out = []
    for p in graph_dir.iterdir():
        m = pat.match(p.name)
        if m and not p.name.endswith(SUFFIX_SKELETON_EDGE):
            out.append(m.group(1))
    return sorted(out)


def build_sample(graph_dir: Path, floor_id: str, meta: dict | None, stats: CorpusStats) -> ExpansionSample | None:
    """One floor → ExpansionSample or None (reason counted in ``stats``)."""
    stats.n_floors_seen += 1
    tgt_p = graph_dir / f"{floor_id}{SUFFIX_TARGET_EDGE}"
    sk_e, sk_n = graph_dir / f"{floor_id}{SUFFIX_SKELETON_EDGE}", graph_dir / f"{floor_id}{SUFFIX_SKELETON_NODE}"
    if not tgt_p.exists():
        stats.n_missing_target += 1
        return None
    if not (sk_e.exists() and sk_n.exists()):
        stats.n_missing_skeleton += 1
        return None
    try:
        skeleton = load_graph_csv(sk_e, sk_n)
        target = load_target_csv(tgt_p)
    except Exception as exc:  # noqa: BLE001
        stats.n_bad += 1
        logger.warning("%s: %s", floor_id, exc)
        return None
    # skeleton must be a subgraph of the target (prototype preservation is an *invariant* of the data)
    tg = nx.Graph(target.edges())
    tg.add_nodes_from(target.nodes)
    if not set(skeleton.nodes) <= set(tg.nodes) or any(not tg.has_edge(u, v) for u, v in skeleton.edges()):
        stats.n_skeleton_not_subgraph += 1
        return None
    if not nx.is_connected(tg):
        stats.n_target_disconnected += 1  # kept, but counted (the evaluator penalises disconnected outputs)
    if target.num_nodes <= skeleton.num_nodes:
        stats.n_no_growth += 1
        return None
    # carry skeleton positions into the target (new nodes have no coordinates in the export)
    target.positions = dict(skeleton.positions)
    target.node_attrs = {**{n: {} for n in target.nodes}, **skeleton.node_attrs}
    meta = meta or {}
    lt = meta.get("layout_type")
    try:
        layout = LayoutType(lt) if lt else None
    except ValueError:
        layout = LayoutType.UNKNOWN
    area = meta.get("total_area")
    area = float(area) if area is not None and not (isinstance(area, float) and np.isnan(area)) and float(area) > 0 else None
    stats.n_written += 1
    return ExpansionSample(
        sample_id=floor_id,
        city=meta.get("cityname"),
        layout_type=layout,
        area_sqm=area,
        target_num_nodes=target.num_nodes,
        skeleton=skeleton,
        target=target,
    )


def _grouped_split(mall_ids: list[str], strata: list[str | None], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """Mall-level split stratified by (city cluster) → {mall_id: split}. Identical malls never straddle splits."""
    rng = np.random.RandomState(seed)
    by_stratum: dict[str | None, list[str]] = {}
    for m, s in zip(mall_ids, strata, strict=True):
        by_stratum.setdefault(s, []).append(m)
    out: dict[str, str] = {}
    for _, malls in sorted(by_stratum.items(), key=lambda kv: str(kv[0])):
        malls = sorted(set(malls))
        rng.shuffle(malls)
        n = len(malls)
        n_test, n_val = int(round(n * test_ratio)), int(round(n * val_ratio))
        for i, m in enumerate(malls):
            out[m] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out


def build_corpus(
    graph_dir: str | Path,
    out_path: str | Path,
    main_table: pd.DataFrame | None = None,
    *,
    id_col: str = "floor_id",
    mall_id_col: str = "mall_id",
    cluster_col: str = "city_cluster",
    layout_col: str = "layout_type",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    limit: int | None = None,
    floor_ids: Iterable[str] | None = None,
) -> tuple[Path, CorpusStats]:
    """Scan ``graph_dir``, build all samples, assign mall-grouped splits, write JSONL. Returns (path, stats)."""
    graph_dir = Path(graph_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(floor_ids) if floor_ids is not None else floor_ids_in(graph_dir)
    if limit:
        ids = ids[:limit]
    meta_by_id: dict[str, dict] = {}
    if main_table is not None:
        cols = [c for c in [id_col, mall_id_col, cluster_col, layout_col, "cityname", "total_area", "total_score"] if c in main_table.columns]
        mt = main_table[cols].copy()
        mt[id_col] = mt[id_col].astype(str)
        meta_by_id = {r[id_col]: {k: (None if pd.isna(v) else v) for k, v in r.items()} for r in mt.to_dict("records")}
    stats = CorpusStats()
    samples: list[tuple[ExpansionSample, dict]] = []
    for fid in ids:
        meta = meta_by_id.get(fid)
        smp = build_sample(graph_dir, fid, meta, stats)
        if smp is not None:
            samples.append((smp, meta or {}))
    malls = [split_floor_id(s.sample_id)[0] for s, _ in samples]
    strata = [str(m.get(cluster_col)) if m.get(cluster_col) is not None else None for _, m in samples]
    split_of = _grouped_split(malls, strata, val_ratio, test_ratio, seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for (smp, meta), mall in zip(samples, malls, strict=True):
            rec = smp.to_record()
            rec.update({
                "mall_id": mall,
                "split": split_of[mall],
                "city_cluster": meta.get(cluster_col),
                "total_score": meta.get("total_score"),
                "skeleton_positions": {k: [float(x), float(y)] for k, (x, y) in smp.skeleton.positions.items()},
                "target_edge_attrs": smp.target.edge_attrs,
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Stage-2 corpus v2: %d samples → %s  (%s)", len(samples), out_path, stats.as_dict())
    return out_path, stats


# --------------------------------------------------------------------------- loading
def load_corpus_jsonl(path: str | Path, split: str | None = None, limit: int | None = None) -> list[ExpansionSample]:
    """Read the JSONL corpus. ``split`` ∈ {train, val, test, None=all}. Positions / edge attrs restored."""
    out: list[ExpansionSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if split and rec.get("split") != split:
                continue
            smp = ExpansionSample.from_record(rec)
            pos = {k: (float(v[0]), float(v[1])) for k, v in (rec.get("skeleton_positions") or {}).items()}
            smp.skeleton.positions = pos
            smp.target.positions = dict(pos)
            smp.target.edge_attrs = rec.get("target_edge_attrs") or {}
            smp.mall_id = rec.get("mall_id")
            smp.split = rec.get("split")
            out.append(smp)
            if limit and len(out) >= limit:
                break
    return out


def load_any_corpus(path: str | Path, split: str | None = None, limit: int | None = None) -> list[ExpansionSample]:
    """Dispatch on file type: ``.jsonl`` → corpus v2 (real CSV-derived); ``.json`` → legacy ShareGPT (v1).

    For the legacy format ``split`` is emulated by position (last ``eval_holdout`` records are the
    test tail) by the callers, so it is ignored here.
    """
    from mall_space_planner.data.sharegpt_adapter import load_sharegpt

    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return load_corpus_jsonl(p, split=split, limit=limit)
    return load_sharegpt(p, limit=limit)


__all__ = ["CorpusStats", "build_corpus", "build_sample", "floor_ids_in", "load_any_corpus", "load_corpus_jsonl", "load_target_csv", "parse_centerpoint"]
