"""Abstract interfaces for Stage-2 components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mall_space_planner.schemas import (
    ConstraintSet,
    GeneratedLayout,
    SiteBoundary,
    TopologyGraph,
    TopologyPrototype,
)


@dataclass
class GenerationRequest:
    """Inputs to Stage 2."""

    prototype: TopologyPrototype
    boundary: SiteBoundary
    constraints: ConstraintSet
    n_candidates: int = 1
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTopologyGenerator(ABC):
    """Expand a prototype skeleton to a full topology of ~N_target nodes."""

    @abstractmethod
    def generate(self, request: GenerationRequest, seed: int) -> TopologyGraph: ...

    def fit(self, samples: list[Any]) -> BaseTopologyGenerator:  # learned generators override
        return self


class BaseGeometryDecoder(ABC):
    """Embed a topology in 2-D inside the boundary and produce space units."""

    @abstractmethod
    def decode(self, topology: TopologyGraph, request: GenerationRequest, seed: int) -> GeneratedLayout: ...


class BaseConstraintSolver(ABC):
    @abstractmethod
    def solve(self, layout: GeneratedLayout, constraints: ConstraintSet) -> GeneratedLayout: ...


class BaseRepairer(ABC):
    @abstractmethod
    def repair(self, layout: GeneratedLayout, request: GenerationRequest) -> GeneratedLayout: ...
