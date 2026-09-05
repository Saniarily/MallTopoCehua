"""Core data contracts.

Design notes
------------
* Everything is plain Pydantic v2 so that objects validate on construction, serialise
  to JSON for the API / UI, and can be round-tripped through ``data/processed``.
* Graph objects are kept framework-agnostic (adjacency lists + optional coordinates);
  conversion to ``networkx`` / PyG happens in :mod:`mall_space_planner.topology`.
* All numeric planning features are optional (``None`` = missing) so that the same
  schema serves both the real legacy table and partially-specified user queries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LayoutType(str, Enum):
    """Corridor-skeleton typology used in the Stage-2 corpus (6 classes + unknown).

    Chinese labels are the ones that appear verbatim in ``sharegpt_data.json``.
    """

    LINEAR = "一字型"
    SIMPLE_LOOP = "简单环型"
    MULTI_LOOP = "多环型"
    SIMPLE_CENTRAL = "简单集中型"
    COMPLEX_CENTRAL = "复杂集中型"
    SIMPLE = "简单型"
    UNKNOWN = "Unknown_Layout"


LAYOUT_TYPES: tuple[str, ...] = tuple(t.value for t in LayoutType)


# --------------------------------------------------------------------------- planning
class PlanningCondition(BaseModel):
    """Planning-side query: city economics, site scale and competition context.

    Field names mirror the legacy main table (``config.yaml -> features.query_cols``).
    Units were **not** documented in the legacy repo; the values below are recorded as
    observed in the legacy ``scaler.pkl`` (mean/std over 3 983 training floors) and
    should be confirmed with the data owner (see ``docs/data_audit.md``).
    """

    model_config = ConfigDict(extra="allow")

    city_cluster: int | None = Field(None, description="Categorical city tier/cluster id (hard filter in legacy).")
    city_name: str | None = Field(None, description="Optional city name (present in Stage-2 corpus).")

    people: float | None = Field(None, description="City population (legacy mean≈9.6e3; unit unconfirmed, likely 1e3).")
    GDP_2023: float | None = Field(None, description="City GDP 2023 (legacy mean≈5.8e4).")
    PCDI_2023: float | None = Field(None, description="Per-capita disposable income 2023 (legacy mean≈5.6e4).")
    TP_2023: float | None = Field(None, description="Total retail/consumption indicator 2023 (legacy mean≈5.0e3).")
    mall_area_count: float | None = Field(None, description="Number of malls in the city/area (legacy mean≈101).")
    nearest_distance_km: float | None = Field(None, ge=0, description="Distance to nearest mall in km (mean≈0.22).")
    count_1km: float | None = Field(None, ge=0, description="Malls within 1 km (mean≈11).")
    count_2km: float | None = Field(None, ge=0, description="Malls within 2 km (mean≈28).")
    total_area: float | None = Field(None, gt=0, description="Project total floor area in m² (mean≈1.2e5).")
    Tx: float | None = Field(None, description="Legacy feature 'Tx' (mean≈27.7, std≈16.7). Semantics unconfirmed.")

    preferred_layout: LayoutType | None = Field(None, description="Optional user preference for skeleton type.")

    def feature_vector(self, cols: list[str]) -> list[float | None]:
        """Return values for ``cols`` in order (``None`` for missing)."""
        data = self.model_dump()
        return [data.get(c) for c in cols]


# --------------------------------------------------------------------------- geometry
class SiteBoundary(BaseModel):
    """Planar site outline (exterior ring + optional holes) in metres."""

    exterior: list[tuple[float, float]] = Field(..., min_length=3)
    holes: list[list[tuple[float, float]]] = Field(default_factory=list)
    crs: str | None = Field(None, description="Optional CRS/EPSG string; None = local metric frame.")
    entrances: list[tuple[float, float]] = Field(default_factory=list, description="User-marked entrance points.")
    atrium_hints: list[tuple[float, float]] = Field(default_factory=list, description="Optional preferred atrium centres.")

    @field_validator("exterior")
    @classmethod
    def _close_ring_not_required(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        # Accept both closed and open rings; drop duplicate closing vertex.
        if len(v) >= 4 and v[0] == v[-1]:
            v = v[:-1]
        if len(v) < 3:
            raise ValueError("exterior ring needs at least 3 distinct vertices")
        return v

    def area(self) -> float:
        """Polygon area (shoelace) minus holes; no shapely dependency here."""

        def ring_area(r: list[tuple[float, float]]) -> float:
            s = 0.0
            for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1], strict=True):
                s += x1 * y2 - x2 * y1
            return abs(s) / 2.0

        return ring_area(self.exterior) - sum(ring_area(h) for h in self.holes)

    @classmethod
    def rectangle(cls, width: float, height: float, origin: tuple[float, float] = (0.0, 0.0)) -> SiteBoundary:
        x0, y0 = origin
        return cls(exterior=[(x0, y0), (x0 + width, y0), (x0 + width, y0 + height), (x0, y0 + height)])


# --------------------------------------------------------------------------- topology
class TopologyGraph(BaseModel):
    """Undirected graph as adjacency list with optional per-node attributes.

    ``adjacency`` maps node id -> sorted list of neighbour ids. The validator enforces
    symmetry (adds missing reverse edges) and removes self-loops, mirroring how the
    legacy loader treated edge CSVs as undirected.
    """

    model_config = ConfigDict(extra="forbid")

    adjacency: dict[str, list[str]]
    node_types: dict[str, str] = Field(default_factory=dict, description="e.g. M=corridor junction, L=corridor, A/D/E/G=shop types.")
    positions: dict[str, tuple[float, float]] = Field(default_factory=dict, description="Optional 2-D coordinates.")
    node_attrs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    edge_attrs: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Keyed by 'u|v' with u<v.")

    @model_validator(mode="after")
    def _symmetrise(self) -> TopologyGraph:
        adj: dict[str, set[str]] = {str(k): set() for k in self.adjacency}
        for u, nbrs in self.adjacency.items():
            for v in nbrs:
                u_s, v_s = str(u), str(v)
                if u_s == v_s:
                    continue
                adj.setdefault(u_s, set()).add(v_s)
                adj.setdefault(v_s, set()).add(u_s)
        self.adjacency = {k: sorted(v) for k, v in sorted(adj.items())}
        return self

    # convenience -----------------------------------------------------------------
    @property
    def nodes(self) -> list[str]:
        return list(self.adjacency)

    @property
    def num_nodes(self) -> int:
        return len(self.adjacency)

    def edges(self) -> list[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for u, nbrs in self.adjacency.items():
            for v in nbrs:
                out.add((u, v) if u < v else (v, u))
        return sorted(out)

    @property
    def num_edges(self) -> int:
        return len(self.edges())

    def degree(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.adjacency.items()}

    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[str, str]],
        nodes: list[str] | None = None,
        positions: dict[str, tuple[float, float]] | None = None,
    ) -> TopologyGraph:
        adj: dict[str, list[str]] = {str(n): [] for n in (nodes or [])}
        for u, v in edges:
            adj.setdefault(str(u), []).append(str(v))
            adj.setdefault(str(v), []).append(str(u))
        return cls(adjacency=adj, positions=positions or {})


class TopologyMetrics(BaseModel):
    """Hand-crafted graph descriptors (subset kept identical to legacy ``metric_cols``).

    Legacy provided ``L1_density, L2_diameter, L2_complexity, L2_integration`` in the main
    table; the remaining fields are recomputed by :mod:`mall_space_planner.topology.metrics`.
    """

    model_config = ConfigDict(extra="allow")

    num_nodes: int | None = None
    num_edges: int | None = None
    density: float | None = None
    avg_degree: float | None = None
    diameter: int | None = None
    avg_shortest_path: float | None = None
    num_cycles: int | None = None
    n_components: int | None = None
    clustering: float | None = None
    degree_entropy: float | None = None
    max_betweenness: float | None = None
    # Legacy pre-computed columns (names kept verbatim for traceability)
    L1_density: float | None = None
    L2_diameter: float | None = None
    L2_complexity: float | None = None
    L2_integration: float | None = None


class TopologyPrototype(BaseModel):
    """A candidate corridor-junction skeleton drawn from the case database.

    In the current data every *floor* (``floor_id``) with an ``_M_simplified`` graph is a
    prototype; a ``prototype_id`` is therefore the floor id. Clustering into canonical
    prototypes is a future option (see ``docs/methodology.md``).
    """

    prototype_id: str
    graph: TopologyGraph
    layout_type: LayoutType | None = None
    metrics: TopologyMetrics = Field(default_factory=TopologyMetrics)
    source_case_id: str | None = Field(None, description="mall_id or floor_id this prototype came from.")
    quality_score: float | None = Field(None, description="Legacy 'total_score' (higher = better).")


class MallCase(BaseModel):
    """One row of the case database: a floor plan with its context and prototype."""

    model_config = ConfigDict(extra="allow")

    floor_id: str
    mall_id: str
    floor_index: int | None = None
    condition: PlanningCondition
    prototype: TopologyPrototype | None = None
    total_score: float | None = None
    boundary: SiteBoundary | None = Field(None, description="Not available in legacy data (see audit).")


class RankingLabel(BaseModel):
    """Supervision for Stage 1: for a query, graded relevance of a candidate."""

    query_id: str
    candidate_id: str
    relevance: float = Field(..., description="Graded relevance (e.g. min-max normalised total_score within group).")
    group_id: str | None = Field(None, description="Bucket used to form the candidate list (city_cluster x area bin).")


# --------------------------------------------------------------------------- recommendations
class RecommendationExplanation(BaseModel):
    recommendation_summary: str
    top_factors: list[dict[str, Any]] = Field(default_factory=list)
    matched_case_evidence: list[dict[str, Any]] = Field(default_factory=list)
    topology_reasoning: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None


class Recommendation(BaseModel):
    rank: int
    prototype_id: str
    score: float
    confidence: float | None = None
    quality_score: float | None = None
    similarity: float | None = None
    layout_type: LayoutType | None = None
    metrics: TopologyMetrics | None = None
    explanation: RecommendationExplanation | None = None


# --------------------------------------------------------------------------- stage 2
class ConstraintSet(BaseModel):
    """User constraints for Stage-2 generation, with weights for soft terms."""

    target_num_nodes: int | None = Field(None, ge=2, description="Target total node count (N_target).")
    target_num_shops: int | None = Field(None, ge=0)
    shop_area_min: float | None = Field(None, gt=0, description="m²")
    shop_area_max: float | None = Field(None, gt=0, description="m²")
    shop_area_mean: float | None = Field(None, gt=0)
    corridor_width: float = Field(6.0, gt=0, description="Nominal corridor width in metres.")
    min_entrances: int = Field(1, ge=0)
    num_atria: int | None = Field(None, ge=0)
    target_metrics: TopologyMetrics | None = Field(None, description="Optional target graph descriptors.")
    preserve_prototype_edges: bool = True
    layout_type: LayoutType | None = None
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "topology": 1.0,
            "prototype": 1.0,
            "boundary": 1.0,
            "overlap": 1.0,
            "area": 0.5,
            "connectivity": 1.0,
        }
    )

    @model_validator(mode="after")
    def _check_area_range(self) -> ConstraintSet:
        if self.shop_area_min and self.shop_area_max and self.shop_area_min > self.shop_area_max:
            raise ValueError("shop_area_min must be <= shop_area_max")
        return self


class SpaceUnit(BaseModel):
    """A 2-D space unit produced by Stage 2 (shop, corridor segment, atrium, entrance)."""

    unit_id: str
    kind: Literal["shop", "corridor", "junction", "atrium", "entrance"]
    polygon: list[tuple[float, float]] | None = None
    centroid: tuple[float, float] | None = None
    area: float | None = None
    attached_to: list[str] = Field(default_factory=list, description="Ids of connected units / graph nodes.")
    attrs: dict[str, Any] = Field(default_factory=dict)


class GeneratedLayout(BaseModel):
    """Full Stage-2 output: expanded topology + geometry + diagnostics."""

    layout_id: str
    prototype_id: str | None = None
    boundary: SiteBoundary
    topology: TopologyGraph
    skeleton_positions: dict[str, tuple[float, float]] = Field(default_factory=dict)
    units: list[SpaceUnit] = Field(default_factory=list)
    constraints: ConstraintSet | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    generator_name: str | None = None
    seed: int | None = None


class EvaluationResult(BaseModel):
    """Generic evaluation container used by Stage-1 and Stage-2 evaluators."""

    evaluator: str
    metrics: dict[str, float | int | None]
    passed: dict[str, bool] = Field(default_factory=dict)
    overall_pass: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)
