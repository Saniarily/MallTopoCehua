"""Rule-based skeleton expansion (stable, non-neural Stage-2 baseline).

Given a prototype skeleton ``G_in`` and target size ``N_target``, the expander adds new
nodes by three operations chosen stochastically with configurable probabilities, while
**never removing a skeleton edge** (edge accuracy = 100 % by construction):

* ``subdivide``  – insert a node on an existing edge (u–v → u–w–v). Preserving the
  skeleton edge is required by the spec, so the original edge is kept **and** the new node
  is attached to both endpoints (forming a triangle), which mimics side corridors.
* ``branch``     – attach a degree-1 node (dead-end corridor / anchor shop) to a node,
  preferring low-degree skeleton nodes.
* ``chord``      – connect two existing nodes at graph distance 2–3 to create a loop
  (only for loop-type layouts by default). Since corpus v2 showed that real expansions never add
  an edge between two *skeleton* nodes, chords whose both endpoints are skeleton nodes are skipped.
* ``bridge``     – add a node connected to **two** existing nodes at graph distance 2–4 (a new
  corridor segment closing a loop through the new key-point) – the dominant growth pattern in the
  real corridor topologies (≈ 75 % of new nodes have ≥ 2 anchors).

``planar_guard`` (default on) rejects operations that would make the graph non-planar.

The mixture is tuned so that density and ASPL track the expected baselines of the
evaluation spec: the target average degree is kept close to the skeleton's average degree.
When ``layout_type`` is given, operation probabilities are adjusted (e.g. linear → more
branches, multi-loop → more chords).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from mall_space_planner.data.synthetic import letter_label
from mall_space_planner.registry import register
from mall_space_planner.schemas import LayoutType, TopologyGraph
from mall_space_planner.stage2.base import BaseTopologyGenerator, GenerationRequest
from mall_space_planner.topology.convert import from_networkx, to_networkx

_LAYOUT_PRIORS: dict[LayoutType | None, dict[str, float]] = {
    None: {"subdivide": 0.30, "branch": 0.30, "chord": 0.10, "bridge": 0.30},
    LayoutType.LINEAR: {"subdivide": 0.35, "branch": 0.40, "chord": 0.05, "bridge": 0.20},
    LayoutType.SIMPLE: {"subdivide": 0.30, "branch": 0.45, "chord": 0.05, "bridge": 0.20},
    LayoutType.SIMPLE_LOOP: {"subdivide": 0.35, "branch": 0.30, "chord": 0.10, "bridge": 0.25},
    LayoutType.MULTI_LOOP: {"subdivide": 0.30, "branch": 0.20, "chord": 0.15, "bridge": 0.35},
    LayoutType.SIMPLE_CENTRAL: {"subdivide": 0.30, "branch": 0.40, "chord": 0.05, "bridge": 0.25},
    LayoutType.COMPLEX_CENTRAL: {"subdivide": 0.30, "branch": 0.25, "chord": 0.15, "bridge": 0.30},
    LayoutType.UNKNOWN: {"subdivide": 0.30, "branch": 0.30, "chord": 0.10, "bridge": 0.30},
}
_OP_NAMES = ["subdivide", "branch", "chord", "bridge"]


@dataclass
class _Ops:
    subdivide: float
    branch: float
    chord: float
    bridge: float = 0.0

    def normalised(self) -> np.ndarray:
        v = np.array([self.subdivide, self.branch, self.chord, self.bridge], dtype=float)
        return v / v.sum()


def _new_label(g: nx.Graph, style: str, counter: int) -> str:
    if style == "letters":
        i = counter
        while letter_label(i) in g:
            i += 1
        return letter_label(i)
    i = counter
    while f"N{i:03d}" in g:
        i += 1
    return f"N{i:03d}"


@register("generator", "rule_expander")
class RuleBasedExpander(BaseTopologyGenerator):
    def __init__(
        self,
        op_probs: dict[str, float] | None = None,
        target_degree_tolerance: float = 0.15,
        label_style: str = "letters",
        max_iters_factor: int = 20,
        planar_guard: bool = True,
    ) -> None:
        self.op_probs = op_probs
        self.target_degree_tolerance = target_degree_tolerance
        self.label_style = label_style
        self.max_iters_factor = max_iters_factor
        self.planar_guard = planar_guard

    # ------------------------------------------------------------------ helpers
    def _ops(self, layout: LayoutType | None) -> _Ops:
        base = dict(_LAYOUT_PRIORS.get(layout, _LAYOUT_PRIORS[None]))
        if self.op_probs:
            base.update(self.op_probs)
        return _Ops(**base)

    @staticmethod
    def _avg_degree(g: nx.Graph) -> float:
        return 2.0 * g.number_of_edges() / max(1, g.number_of_nodes())

    def _pick_node(self, g: nx.Graph, rng: np.random.RandomState, prefer_low_degree: bool = True) -> str:
        nodes = list(g.nodes)
        deg = np.array([g.degree(n) for n in nodes], dtype=float)
        w = 1.0 / (deg + 1.0) if prefer_low_degree else deg + 1.0
        w = w / w.sum()
        return nodes[int(rng.choice(len(nodes), p=w))]

    # ------------------------------------------------------------------ main
    def generate(self, request: GenerationRequest, seed: int) -> TopologyGraph:
        rng = np.random.RandomState(seed)
        skeleton = request.prototype.graph
        g = to_networkx(skeleton)
        skeleton_edges = set(skeleton.edges())
        n_target = request.constraints.target_num_nodes or int(round(skeleton.num_nodes * 1.5))
        n_target = max(n_target, skeleton.num_nodes)
        layout = request.constraints.layout_type or request.prototype.layout_type
        probs = self._ops(layout).normalised()
        ref_deg = self._avg_degree(g)
        sk_nodes = set(skeleton.nodes)

        def _try_add(w: str, anchors: list[str]) -> bool:
            g.add_node(w)
            g.add_edges_from((a, w) for a in anchors)
            if self.planar_guard and len(anchors) >= 2 and not nx.check_planarity(g)[0]:
                g.remove_node(w)
                return False
            return True

        counter = g.number_of_nodes()
        iters = 0
        max_iters = self.max_iters_factor * max(1, n_target)
        while g.number_of_nodes() < n_target and iters < max_iters:
            iters += 1
            # Degree control: if we are denser than the reference, prefer branches (add degree-1 nodes);
            # if sparser, prefer subdivisions / chords.
            cur = self._avg_degree(g)
            p = probs.copy()
            if cur > ref_deg * (1 + self.target_degree_tolerance):
                p = p * np.array([0.5, 2.0, 0.2, 0.4])
            elif cur < ref_deg * (1 - self.target_degree_tolerance):
                p = p * np.array([1.5, 0.5, 1.5, 1.5])
            p = p / p.sum()
            op = _OP_NAMES[int(rng.choice(len(_OP_NAMES), p=p))]

            if op == "subdivide" and g.number_of_edges() > 0:
                edges = list(g.edges)
                u, v = edges[int(rng.randint(len(edges)))]
                w = _new_label(g, self.label_style, counter)
                if _try_add(w, [u, v]):  # keep (u,v) as well → skeleton edge preserved
                    counter += 1
            elif op == "branch":
                anchor = self._pick_node(g, rng, prefer_low_degree=True)
                w = _new_label(g, self.label_style, counter)
                counter += 1
                g.add_node(w)
                g.add_edge(anchor, w)
            elif op == "bridge":  # new node closing a loop between two nodes at distance 2..4
                u = self._pick_node(g, rng, prefer_low_degree=False)
                lengths = nx.single_source_shortest_path_length(g, u, cutoff=4)
                cands = [v for v, d in lengths.items() if 2 <= d <= 4]
                w = _new_label(g, self.label_style, counter)
                if cands and _try_add(w, [u, cands[int(rng.randint(len(cands)))]]):
                    counter += 1
                else:
                    g.add_node(w)
                    g.add_edge(u, w)
                    counter += 1
            else:  # chord: connect nodes at distance 2..3 (does not add a node; bounded)
                if g.number_of_nodes() < 4:
                    continue
                u = self._pick_node(g, rng, prefer_low_degree=False)
                lengths = nx.single_source_shortest_path_length(g, u, cutoff=3)
                # never add a skeleton–skeleton edge (violates the corpus invariant: prototype kept verbatim)
                cands = [v for v, d in lengths.items() if 2 <= d <= 3 and not g.has_edge(u, v) and not (u in sk_nodes and v in sk_nodes)]
                if cands:
                    v = cands[int(rng.randint(len(cands)))]
                    g.add_edge(u, v)
                    if self.planar_guard and not nx.check_planarity(g)[0]:
                        g.remove_edge(u, v)
                # a chord does not increase node count; add a branch to make progress
                anchor = self._pick_node(g, rng, prefer_low_degree=True)
                w = _new_label(g, self.label_style, counter)
                counter += 1
                g.add_node(w)
                g.add_edge(anchor, w)

        # Safety: skeleton edges must all be present.
        for u, v in skeleton_edges:
            if not g.has_edge(u, v):
                g.add_edge(u, v)
        out = from_networkx(g)
        out.node_types = {**{n: skeleton.node_types.get(n, "M") for n in skeleton.nodes}, **{n: "M" for n in out.nodes if n not in skeleton.nodes}}
        return out
