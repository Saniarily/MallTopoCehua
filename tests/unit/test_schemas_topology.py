import math
import pytest
from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyGraph
from mall_space_planner.topology.metrics import aspl_deviation, compute_topology_metrics, density_deviation, edge_accuracy, node_deviation

def test_graph_symmetrise_and_edges():
    g = TopologyGraph(adjacency={"A": ["B", "A"], "B": [], "C": ["B"]})
    assert g.adjacency["B"] == ["A", "C"] and g.edges() == [("A", "B"), ("B", "C")] and g.num_edges == 2

def test_spec_metrics_identity():
    sk = TopologyGraph(adjacency={"A": ["B"], "B": ["C"], "C": ["D"], "D": []})
    assert edge_accuracy(sk, sk) == 100.0 and node_deviation(sk, 4) == 0.0
    assert density_deviation(sk, sk, 4) == pytest.approx(0.0) and aspl_deviation(sk, sk, 4) == pytest.approx(0.0)
    m = compute_topology_metrics(sk); assert m.num_cycles == 0 and m.n_components == 1 and m.diameter == 3

def test_boundary_area_and_constraints():
    assert SiteBoundary.rectangle(10, 5).area() == 50
    with pytest.raises(ValueError):
        ConstraintSet(shop_area_min=100, shop_area_max=50)
    assert math.isclose(SiteBoundary(exterior=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]).area(), 1.0)
