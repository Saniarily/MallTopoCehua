"""Conversions between :class:`TopologyGraph` and ``networkx``."""

from __future__ import annotations

import networkx as nx

from mall_space_planner.schemas import TopologyGraph


def to_networkx(graph: TopologyGraph) -> nx.Graph:
    """Build an undirected ``networkx.Graph`` carrying positions/types as node attributes."""
    g = nx.Graph()
    for n in graph.nodes:
        attrs: dict = dict(graph.node_attrs.get(n, {}))
        if n in graph.positions:
            attrs["pos"] = tuple(graph.positions[n])
        if n in graph.node_types:
            attrs["ntype"] = graph.node_types[n]
        g.add_node(n, **attrs)
    for u, v in graph.edges():
        key = f"{u}|{v}"
        g.add_edge(u, v, **graph.edge_attrs.get(key, {}))
    return g


def from_networkx(g: nx.Graph) -> TopologyGraph:
    """Inverse of :func:`to_networkx`; node ids are stringified."""
    adjacency = {str(n): [str(m) for m in g.neighbors(n)] for n in g.nodes}
    positions = {str(n): tuple(d["pos"]) for n, d in g.nodes(data=True) if "pos" in d}
    node_types = {str(n): d["ntype"] for n, d in g.nodes(data=True) if "ntype" in d}
    node_attrs = {
        str(n): {k: v for k, v in d.items() if k not in ("pos", "ntype")}
        for n, d in g.nodes(data=True)
        if any(k not in ("pos", "ntype") for k in d)
    }
    return TopologyGraph(adjacency=adjacency, positions=positions, node_types=node_types, node_attrs=node_attrs)
