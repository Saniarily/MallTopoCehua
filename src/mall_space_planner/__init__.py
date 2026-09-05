"""MallTopoCehua — mall spatial planning with topology prototypes as an intermediate state.

Two-stage framework:

* **Stage 1** — retrieve, rank and explain topology prototypes given planning conditions.
* **Stage 2** — controllably expand the selected prototype into a full mall topology
  and a draft 2-D layout under site constraints.

The package is organised as a set of pluggable components registered through
:mod:`mall_space_planner.registry`; concrete models are selected purely through YAML
configuration (see ``configs/``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
