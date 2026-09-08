"""Viewer Hub backend (Phase 5): UI-independent services used by the Streamlit app and the FastAPI server.

* :mod:`jobs`     – background job runner for training / evaluation scripts (subprocess, log file, status, stop)
* :mod:`catalog`  – case-database browsing helpers (stats, filters, topology + outline access, anomaly report)
* :mod:`workbench`– planning workbench: Stage 1 → Stage 2 → Stage 3 (new build) and the renovation workflow, with
                    re-generation of individual candidates and export
* :mod:`experiments` – experiment registry over ``outputs/experiments`` (tables, per-seed CSVs, checkpoints)

No plotting or Streamlit code lives here.
"""
