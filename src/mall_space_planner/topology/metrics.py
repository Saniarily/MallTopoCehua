"""Graph descriptors and Stage-2 acceptance metrics.

The five acceptance metrics (node deviation, edge accuracy, density deviation, ASPL
deviation, inference time) implement the definitions in the attached
``评价指标说明.md`` / ``测试大纲及记录`` **verbatim**, including the "expected density"
and "log-scaled expected ASPL" baselines. Thresholds live in configuration
(``configs/stage2/*.yaml``), not in code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import networkx as nx
import numpy as np

from mall_space_planner.schemas import TopologyGraph, TopologyMetrics
from mall_space_planner.topology.convert import to_networkx

EPS = 1e-12


# --------------------------------------------------------------------------- basics
def _largest_cc(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g
    if nx.is_connected(g):
        return g
    comp = max(nx.connected_components(g), key=len)
    return g.subgraph(comp).copy()


def avg_shortest_path(g: nx.Graph) -> float:
    """ASPL on the largest connected component (0 for <=1 node)."""
    if g.number_of_nodes() <= 1:
        return 0.0
    sub = _largest_cc(g)
    if sub.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_shortest_path_length(sub))


def graph_density(g: nx.Graph) -> float:
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    return 2.0 * g.number_of_edges() / (n * (n - 1))


def _edge_set(g: TopologyGraph | nx.Graph) -> set[tuple[str, str]]:
    if isinstance(g, TopologyGraph):
        return set(g.edges())
    return {(str(u), str(v)) if str(u) < str(v) else (str(v), str(u)) for u, v in g.edges()}


def _nx(g: TopologyGraph | nx.Graph) -> nx.Graph:
    return to_networkx(g) if isinstance(g, TopologyGraph) else g


# --------------------------------------------------------------------------- spec metrics
def node_deviation(g_gen: TopologyGraph | nx.Graph, n_target: int) -> float:
    """Metric 1: ``|N_gen − N_target| / N_target × 100`` (percent)."""
    n_gen = _nx(g_gen).number_of_nodes()
    if n_target <= 0:
        return float("nan")
    return abs(n_gen - n_target) / n_target * 100.0


def edge_accuracy(g_in: TopologyGraph | nx.Graph, g_gen: TopologyGraph | nx.Graph) -> float:
    """Metric 2: recall of skeleton edges in the generated graph (percent)."""
    e_in, e_gen = _edge_set(g_in), _edge_set(g_gen)
    if not e_in:
        return 0.0
    return len(e_in & e_gen) / len(e_in) * 100.0


def expected_density(g_in: TopologyGraph | nx.Graph, n_target: int) -> float:
    """``avg_deg_ref / (N_target − 1)`` where ``avg_deg_ref = 2|E_in|/|V_in|``."""
    gi = _nx(g_in)
    if gi.number_of_nodes() == 0 or n_target <= 1:
        return 0.0
    avg_deg_ref = 2.0 * gi.number_of_edges() / gi.number_of_nodes()
    return avg_deg_ref / (n_target - 1)


def density_deviation(g_in: TopologyGraph | nx.Graph, g_gen: TopologyGraph | nx.Graph, n_target: int) -> float:
    """Metric 3: relative deviation from the expected density (percent)."""
    d_exp = expected_density(g_in, n_target)
    if d_exp <= EPS:
        return 0.0
    d_gen = graph_density(_nx(g_gen))
    return abs(d_gen - d_exp) / d_exp * 100.0


def aspl_deviation(g_in: TopologyGraph | nx.Graph, g_gen: TopologyGraph | nx.Graph, n_target: int) -> float:
    """Metric 4: relative deviation from log-scaled expected ASPL (percent)."""
    gi, gg = _nx(g_in), _nx(g_gen)
    aspl_ref = avg_shortest_path(gi)
    n_in = gi.number_of_nodes()
    if aspl_ref <= EPS or n_in <= 1 or n_target <= 1:
        return 0.0
    scale = math.log(n_target) / math.log(n_in) if n_in > 1 else 1.0
    aspl_exp = aspl_ref * scale
    if aspl_exp <= EPS:
        return 0.0
    aspl_gen = avg_shortest_path(gg)
    return abs(aspl_gen - aspl_exp) / aspl_exp * 100.0


# --------------------------------------------------------------------------- descriptors
def _degree_entropy(degrees: Iterable[int]) -> float:
    d = np.asarray(list(degrees), dtype=float)
    if d.size == 0:
        return 0.0
    vals, counts = np.unique(d, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + EPS)).sum())


def compute_topology_metrics(graph: TopologyGraph | nx.Graph, with_betweenness: bool = True) -> TopologyMetrics:
    """Compute the standard descriptor set used across Stage 1 features and Stage 2 evaluation."""
    g = _nx(graph)
    n, m = g.number_of_nodes(), g.number_of_edges()
    if n == 0:
        return TopologyMetrics(num_nodes=0, num_edges=0, n_components=0)
    sub = _largest_cc(g)
    degs = [d for _, d in g.degree()]
    out = TopologyMetrics(
        num_nodes=n,
        num_edges=m,
        density=graph_density(g),
        avg_degree=float(np.mean(degs)) if degs else 0.0,
        diameter=int(nx.diameter(sub)) if sub.number_of_nodes() > 1 else 0,
        avg_shortest_path=avg_shortest_path(g),
        num_cycles=max(0, m - n + nx.number_connected_components(g)),  # cyclomatic number
        n_components=nx.number_connected_components(g),
        clustering=float(nx.average_clustering(g)) if n > 2 else 0.0,
        degree_entropy=_degree_entropy(degs),
    )
    if with_betweenness and n > 2:
        bc = nx.betweenness_centrality(g, normalized=True)
        out.max_betweenness = float(max(bc.values())) if bc else 0.0
    return out


# --------------------------------------------------------------------------- label-free structural comparison
def skeleton_attachment_profile(skeleton: TopologyGraph | nx.Graph, full: TopologyGraph | nx.Graph) -> dict[str, int]:
    """For every skeleton node: number of *new* (non-skeleton) neighbours in ``full``.

    Skeleton labels are shared between generated and target graphs, so this profile compares "which
    skeleton nodes grow branches, and how many" without depending on the labels of new nodes.
    """
    sk, g = _nx(skeleton), _nx(full)
    sk_nodes = set(sk.nodes)
    return {u: sum(1 for v in g.neighbors(u) if v not in sk_nodes) for u in sk_nodes if u in g}


def attachment_overlap(skeleton: TopologyGraph | nx.Graph, target: TopologyGraph | nx.Graph, generated: TopologyGraph | nx.Graph) -> tuple[float, float]:
    """(recall %, precision %) of skeleton→new attachments as a multiset over skeleton nodes (label-free for new nodes)."""
    pt, pg = skeleton_attachment_profile(skeleton, target), skeleton_attachment_profile(skeleton, generated)
    inter = sum(min(pt.get(u, 0), pg.get(u, 0)) for u in set(pt) | set(pg))
    nt, ng = sum(pt.values()), sum(pg.values())
    return (inter / nt * 100.0 if nt else 100.0, inter / ng * 100.0 if ng else 100.0)


def degree_histogram_emd(a: TopologyGraph | nx.Graph, b: TopologyGraph | nx.Graph, max_deg: int = 8) -> float:
    """1-D earth mover's distance between the (normalised) degree histograms of two graphs (label-free)."""
    import numpy as np

    def hist(g: nx.Graph) -> np.ndarray:
        h = np.zeros(max_deg + 1)
        for _, d in g.degree():
            h[min(d, max_deg)] += 1
        return h / max(1, h.sum())

    ha, hb = hist(_nx(a)), hist(_nx(b))
    return float(np.abs(np.cumsum(ha) - np.cumsum(hb)).sum())


def new_new_edge_ratio(skeleton: TopologyGraph | nx.Graph, full: TopologyGraph | nx.Graph) -> float:
    """Share of new edges that connect two *new* nodes (corridor-like growth) vs. new→skeleton."""
    sk_nodes = set(_nx(skeleton).nodes)
    new_edges = [e for e in _edge_set(full) if e not in _edge_set(skeleton)]
    if not new_edges:
        return 0.0
    return sum(1 for u, v in new_edges if u not in sk_nodes and v not in sk_nodes) / len(new_edges)
