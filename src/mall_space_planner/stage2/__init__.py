"""Stage 2 — controllable topology expansion and draft geometric layout under site constraints.

Decoupled design: ``generator`` (graph expansion) → ``geometry_decoder`` (2-D embedding
inside the site boundary) → ``repairer`` (constraint repair) → ``evaluator``.
"""

from mall_space_planner.stage2.base import (
    BaseConstraintSolver,
    BaseGeometryDecoder,
    BaseRepairer,
    BaseTopologyGenerator,
    GenerationRequest,
)

__all__ = [
    "BaseConstraintSolver",
    "BaseGeometryDecoder",
    "BaseRepairer",
    "BaseTopologyGenerator",
    "GenerationRequest",
]
