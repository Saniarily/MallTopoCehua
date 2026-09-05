"""Data access layer.

* :mod:`.legacy_adapter`   – read the legacy MallTopoRanker main table and graph CSVs
  (read-only, no runtime import of the legacy package).
* :mod:`.sharegpt_adapter` – parse the Stage-2 ShareGPT skeleton→topology corpus.
* :mod:`.synthetic`        – generate a small, fully synthetic dataset for smoke tests.
* :mod:`.splits`           – leakage-safe grouped splits (by ``mall_id``).
* :mod:`.audit`            – data audit routines producing markdown/JSON reports.
* :mod:`.case_db`          – in-memory case database used by Stage 1.
"""

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.data.splits import grouped_split

__all__ = ["CaseDatabase", "grouped_split"]
