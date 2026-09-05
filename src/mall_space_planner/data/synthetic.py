"""Synthetic sample data generator.

Produces a *small* dataset in exactly the legacy on-disk formats so that every pipeline
can be exercised without access to the real (private) data:

* ``main_table.csv``  – legacy main-table columns (ids, 10 query cols, 4 metric cols, label)
* ``graphs/{floor_id}_M_simplified.csv`` and ``..._node_attributes.csv``
* ``sharegpt_sample.json`` – Stage-2 skeleton → topology pairs in ShareGPT format

The skeletons are procedurally generated from the six layout archetypes. **This data is
synthetic and clearly labelled as such; no experimental claim may be based on it.**
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from mall_space_planner.data.legacy_adapter import DEFAULT_METRIC_COLS, DEFAULT_QUERY_COLS
from mall_space_planner.schemas import LayoutType
from mall_space_planner.topology.convert import from_networkx
from mall_space_planner.topology.metrics import compute_topology_metrics
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def letter_label(i: int) -> str:
    """0→A, 25→Z, 26→AA … (matches the labelling scheme of the ShareGPT corpus)."""
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = _LETTERS[r] + s
    return s


# --------------------------------------------------------------------------- skeletons
def make_skeleton(layout: LayoutType, n: int, rng: np.random.RandomState) -> nx.Graph:
    """Procedural corridor-junction skeleton with 2-D positions for a given archetype."""
    n = max(3, n)
    g = nx.Graph()
    if layout in (LayoutType.LINEAR, LayoutType.SIMPLE):
        g = nx.path_graph(n)
        pos = {i: (float(i) * 30.0, float(rng.normal(0, 3))) for i in g}
    elif layout == LayoutType.SIMPLE_LOOP:
        g = nx.cycle_graph(n)
        pos = {i: (60 * math.cos(2 * math.pi * i / n), 40 * math.sin(2 * math.pi * i / n)) for i in g}
    elif layout == LayoutType.MULTI_LOOP:
        cols = max(2, int(round(math.sqrt(n / 2))))
        rows = max(2, int(math.ceil(n / cols)))
        g = nx.grid_2d_graph(rows, cols)
        g = nx.convert_node_labels_to_integers(g, label_attribute="rc")
        pos = {i: (d["rc"][1] * 40.0, d["rc"][0] * 30.0) for i, d in g.nodes(data=True)}
        for i in g:
            del g.nodes[i]["rc"]
    elif layout == LayoutType.SIMPLE_CENTRAL:
        g = nx.star_graph(n - 1)
        pos = {0: (0.0, 0.0)}
        for i in range(1, n):
            a = 2 * math.pi * (i - 1) / (n - 1)
            pos[i] = (50 * math.cos(a), 50 * math.sin(a))
    else:  # COMPLEX_CENTRAL: wheel + a few chords
        g = nx.wheel_graph(n)
        pos = {0: (0.0, 0.0)}
        for i in range(1, n):
            a = 2 * math.pi * (i - 1) / (n - 1)
            pos[i] = (50 * math.cos(a), 50 * math.sin(a))
        for _ in range(max(1, n // 5)):
            u, v = rng.choice(range(1, n), 2, replace=False)
            g.add_edge(int(u), int(v))
    nx.set_node_attributes(g, {i: tuple(map(float, p)) for i, p in pos.items()}, "pos")
    return g


def expand_skeleton(skeleton: nx.Graph, n_target: int, rng: np.random.RandomState) -> nx.Graph:
    """Toy 'ground-truth' expansion: attach new nodes as short branches / chords.

    Keeps every skeleton edge (edge recall = 1) like the real corpus.
    """
    g = skeleton.copy()
    next_id = max(g.nodes) + 1
    while g.number_of_nodes() < n_target:
        anchor = int(rng.choice(list(skeleton.nodes)))
        g.add_node(next_id)
        g.add_edge(anchor, next_id)
        if rng.rand() < 0.35 and g.number_of_nodes() > 3:
            other = int(rng.choice([m for m in g.nodes if m not in (anchor, next_id)]))
            g.add_edge(next_id, other)
        next_id += 1
    return g


def relabel_letters(g: nx.Graph) -> nx.Graph:
    mapping = {n: letter_label(i) for i, n in enumerate(sorted(g.nodes))}
    return nx.relabel_nodes(g, mapping)


def relabel_m_nodes(g: nx.Graph) -> nx.Graph:
    mapping = {n: f"M{i + 1:03d}" for i, n in enumerate(sorted(g.nodes))}
    return nx.relabel_nodes(g, mapping)


# --------------------------------------------------------------------------- dataset
def generate_synthetic_dataset(
    out_dir: str | Path,
    n_malls: int = 40,
    max_floors: int = 4,
    n_stage2: int = 120,
    seed: int = 0,
) -> dict[str, Path]:
    """Write a synthetic dataset to ``out_dir`` and return the paths of its parts."""
    out_dir = Path(out_dir)
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    layouts = [lt for lt in LayoutType if lt != LayoutType.UNKNOWN]

    rows: list[dict] = []
    for m in range(n_malls):
        mall_id = f"SYN{m:05d}"
        city_cluster = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        # City-level features (shared by all floors of a mall) — scales follow legacy scaler stats.
        city = {
            "people": float(np.clip(rng.normal(9600, 9700), 500, 60000)),
            "GDP_2023": float(np.clip(rng.normal(57000, 36000), 3000, 250000)),
            "PCDI_2023": float(np.clip(rng.normal(56000, 21000), 15000, 120000)),
            "TP_2023": float(np.clip(rng.normal(5000, 3400), 200, 20000)),
            "mall_area_count": float(np.clip(rng.normal(100, 74), 1, 400)),
            "nearest_distance_km": float(np.clip(abs(rng.normal(0.2, 0.33)), 0.01, 5)),
            "count_1km": float(np.clip(rng.poisson(11), 0, 60)),
            "count_2km": float(np.clip(rng.poisson(28), 0, 120)),
            "total_area": float(np.clip(rng.lognormal(math.log(100000), 0.7), 8000, 900000)),
            "Tx": float(np.clip(rng.normal(27.7, 16.7), 1, 90)),
        }
        n_floors = int(rng.randint(1, max_floors + 1))
        mall_layout = layouts[rng.randint(len(layouts))]
        for fl in range(1, n_floors + 1):
            floor_id = f"{mall_id}_{fl}"
            n_nodes = int(np.clip(rng.normal(19, 9), 4, 60))
            sk = make_skeleton(mall_layout, n_nodes, rng)
            sk = relabel_m_nodes(sk)
            tg = from_networkx(sk)
            met = compute_topology_metrics(tg, with_betweenness=False)
            # Write legacy-format CSVs
            pd.DataFrame(
                [(u, v) for u, v in sk.edges()], columns=["Source", "Target"]
            ).to_csv(graphs_dir / f"{floor_id}_M_simplified.csv", index=False)
            pd.DataFrame(
                {
                    "Node_ID": list(sk.nodes),
                    "Total_L_Neighbors": [int(rng.randint(1, 6)) for _ in sk.nodes],
                    "CenterPoint": [f"({int(d['pos'][0] * 10 + 2000)}, {int(d['pos'][1] * 10 + 1500)})" for _, d in sk.nodes(data=True)],
                }
            ).to_csv(graphs_dir / f"{floor_id}_M_simplified_node_attributes.csv", index=False)

            # Score: noisy function of density/integration + city wealth (so that models have signal)
            base = 60 + 40 * (met.density or 0) * 3 - 0.2 * (met.avg_shortest_path or 0) + 0.0001 * city["PCDI_2023"]
            total_score = float(np.clip(base + rng.normal(0, 4), 0, 100))
            rows.append(
                {
                    "floor_id": floor_id,
                    "mall_id": mall_id,
                    "city_cluster": city_cluster,
                    "layout_type": mall_layout.value,
                    **city,
                    "L1_density": met.density,
                    "L2_diameter": met.diameter,
                    "L2_complexity": 1.0 + (met.num_cycles or 0) / max(1, met.num_nodes or 1),
                    "L2_integration": 1.0 / (1.0 + (met.avg_shortest_path or 1.0)),
                    "total_score": total_score,
                }
            )
    df = pd.DataFrame(rows)
    cols = ["floor_id", "mall_id", "city_cluster", "layout_type", *DEFAULT_QUERY_COLS, *DEFAULT_METRIC_COLS, "total_score"]
    df = df[cols]
    main_csv = out_dir / "main_table.csv"
    df.to_csv(main_csv, index=False)

    # Stage-2 corpus in ShareGPT format
    records = []
    for i in range(n_stage2):
        layout = layouts[rng.randint(len(layouts))]
        n_sk = int(rng.randint(5, 20))
        sk = make_skeleton(layout, n_sk, rng)
        n_target = int(n_sk * rng.uniform(1.3, 2.2))
        full = expand_skeleton(sk, n_target, rng)
        sk_l, full_l = relabel_letters(sk), relabel_letters(full)
        sk_adj = {n: sorted(sk_l.neighbors(n)) for n in sorted(sk_l.nodes)}
        full_adj = {n: sorted(full_l.neighbors(n)) for n in sorted(full_l.nodes)}
        area = float(np.clip(rng.lognormal(math.log(80000), 0.6), 5000, 800000))
        human = (
            "# Context: Commercial Floor Plan Design\n"
            f"# City: 合成市, Layout: {layout.value}, Area:about {area:.1f} sqm\n"
            f"# Target_Scale: Approx {n_target} nodes\n"
            "# Task: Expand skeleton_graph into complete_topology.\n\n"
            "# Skeleton Input (Core Structure)\n"
            f"skeleton_graph = {json.dumps(sk_adj, ensure_ascii=False, indent=4)}\n\n"
            "# Generated Complete Topology (JSON format)\ncomplete_topology = "
        )
        gpt = "```json\n" + json.dumps(full_adj, ensure_ascii=False, indent=4) + "\n```"
        records.append({"conversations": [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}]})
    sg_path = out_dir / "sharegpt_sample.json"
    with open(sg_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    meta = {
        "synthetic": True,
        "seed": seed,
        "n_malls": n_malls,
        "n_floors": len(df),
        "n_stage2": n_stage2,
        "note": "Synthetic data for smoke tests only. Do not report results on it as experimental findings.",
    }
    with open(out_dir / "META.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Synthetic dataset written to %s (%d floors, %d stage-2 pairs)", out_dir, len(df), n_stage2)
    return {"main_table": main_csv, "graph_dir": graphs_dir, "sharegpt": sg_path, "meta": out_dir / "META.json"}


__all__ = ["generate_synthetic_dataset", "make_skeleton", "expand_skeleton", "letter_label"]
