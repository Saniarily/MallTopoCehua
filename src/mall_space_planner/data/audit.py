"""Data audit routines.

Two entry points:

* :func:`audit_legacy_repo` — read-only inspection of the legacy *repository* (tree,
  config, cached artefacts, ablation summaries). Works even when the raw dataset is not
  present on the machine (as in CI), because the legacy repo ships a PyG graph cache and
  a fitted scaler that carry a lot of information.
* :func:`audit_case_database` — statistics, missingness, duplicates, label distribution
  and leakage checks on a built :class:`CaseDatabase` (real or synthetic).

Both return plain dictionaries and can render Markdown via :func:`render_markdown`.
Nothing here writes into the legacy repository.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mall_space_planner.data.case_db import CaseDatabase
from mall_space_planner.utils.config import load_yaml
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- legacy repo
def _tree_summary(root: Path, max_depth: int = 2, ignore: tuple[str, ...] = (".git", "__pycache__")) -> list[str]:
    lines: list[str] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        dirnames[:] = sorted(d for d in dirnames if d not in ignore)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        name = rel.name if rel.parts else root.name
        n_files = len(filenames)
        lines.append(f"{indent}{name}/  ({n_files} files)")
        if depth < max_depth:
            for fn in sorted(filenames)[:15]:
                lines.append(f"{indent}  {fn}")
            if n_files > 15:
                lines.append(f"{indent}  ... (+{n_files - 15} more)")
    return lines


def audit_legacy_repo(legacy_root: str | Path) -> dict[str, Any]:
    """Collect facts about the legacy repository without modifying it."""
    root = Path(legacy_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    out: dict[str, Any] = {"legacy_root": str(root)}
    out["tree"] = _tree_summary(root)

    cfg_path = root / "config.yaml"
    if cfg_path.exists():
        cfg = load_yaml(cfg_path)
        out["config"] = cfg
        data = cfg.get("data", {})
        out["data_paths"] = {
            "main_table_csv": data.get("main_table_csv"),
            "main_table_exists": bool(data.get("main_table_csv") and Path(str(data["main_table_csv"])).expanduser().exists()),
            "graph_dir": data.get("graph_dir"),
            "graph_dir_exists": bool(data.get("graph_dir") and Path(str(data["graph_dir"])).expanduser().exists()),
        }

    req = root / "requirements.txt"
    if req.exists():
        out["requirements"] = [line.strip() for line in req.read_text(encoding="utf-8").splitlines() if line.strip()]

    src = root / "src"
    if src.exists():
        out["source_files"] = {p.name: sum(1 for _ in p.open(encoding="utf-8", errors="ignore")) for p in sorted(src.glob("*.py"))}

    # PyG cache (ids only — no torch needed)
    cache_dir = root / "cache" / "graphs_pt"
    if cache_dir.exists():
        ids = sorted(p.stem for p in cache_dir.glob("*.pt"))
        malls = Counter(i.rsplit("_", 1)[0] for i in ids)
        floors = Counter(int(i.rsplit("_", 1)[1]) for i in ids if i.rsplit("_", 1)[-1].isdigit())
        out["graph_cache"] = {
            "n_graphs": len(ids),
            "n_malls": len(malls),
            "floors_per_mall": dict(sorted(Counter(malls.values()).items())),
            "floor_index_hist": dict(sorted(floors.items())),
            "example_ids": ids[:5],
        }
        out["graph_cache"].update(_graph_cache_stats(cache_dir))

    # scaler
    scaler_path = root / "cache" / "scaler.pkl"
    if scaler_path.exists():
        out["scaler"] = _scaler_stats(scaler_path, cfg if cfg_path.exists() else {})

    # training log
    log = root / "outputs" / "train_logs" / "epoch_metrics.csv"
    if log.exists():
        df = pd.read_csv(log)
        out["train_log"] = {
            "epochs": int(len(df)),
            "columns": df.columns.tolist(),
            "first": df.iloc[0].to_dict(),
            "best_val_ndcg10": float(df["val_ndcg@10"].max()) if "val_ndcg@10" in df else None,
            "best_epoch": int(df.loc[df["val_ndcg@10"].idxmax(), "epoch"]) if "val_ndcg@10" in df else None,
            "last": df.iloc[-1].to_dict(),
        }

    # ablation summaries
    abl_root = root / "outputs" / "ablation"
    if abl_root.exists():
        rows = []
        for f in sorted(abl_root.glob("*/summaries/summary_all.csv")):
            try:
                d = pd.read_csv(f)
            except Exception:  # noqa: BLE001
                continue
            d = d[d.get("best_epoch", pd.Series(dtype=int)) > 0] if "best_epoch" in d else d
            if len(d):
                d["run"] = f.parent.parent.name
                rows.append(d)
        if rows:
            allr = pd.concat(rows, ignore_index=True)
            out["ablations"] = {
                "n_completed_runs": int(len(allr)),
                "best_ndcg10_overall": float(allr["best_ndcg10"].max()),
                "median_ndcg10": float(allr["best_ndcg10"].median()),
                "median_pair_acc": float(allr["best_pair_acc"].median()) if "best_pair_acc" in allr else None,
                "by_factor": allr.groupby(["factor", "value"])["best_ndcg10"].agg(["mean", "std", "count"]).reset_index().to_dict("records"),
            }
    return out


def _graph_cache_stats(cache_dir: Path) -> dict[str, Any]:
    """Node/edge statistics of the legacy PyG cache (requires torch; skipped otherwise)."""
    try:
        import torch  # noqa: WPS433
        import torch_geometric  # noqa: F401
    except ImportError:
        return {"note": "torch/torch_geometric not installed; node/edge stats skipped"}
    ns, es = [], []
    for p in sorted(cache_dir.glob("*.pt")):
        try:
            d = torch.load(p, weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        ns.append(int(d.num_nodes))
        es.append(int(d.edge_index.shape[1] // 2))
    if not ns:
        return {}
    ns_a, es_a = np.array(ns), np.array(es)
    return {
        "node_feature_dim": 3,
        "node_features": ["Total_L_Neighbors", "x_norm", "y_norm"],
        "nodes": {"min": int(ns_a.min()), "p25": float(np.percentile(ns_a, 25)), "median": float(np.median(ns_a)), "p75": float(np.percentile(ns_a, 75)), "max": int(ns_a.max()), "mean": float(ns_a.mean())},
        "edges": {"min": int(es_a.min()), "median": float(np.median(es_a)), "max": int(es_a.max()), "mean": float(es_a.mean())},
    }


def _scaler_stats(path: Path, cfg: dict) -> dict[str, Any]:
    """Read mean/std of features from the legacy fitted StandardScalers (via a stub class)."""
    import pickle
    import sys
    import types

    # The pickle references `src.utils_scaler.DualScaler`; provide a stand-in so we do
    # not import the legacy package.
    stub_pkg = types.ModuleType("src")
    stub_mod = types.ModuleType("src.utils_scaler")

    class DualScaler:  # noqa: D401 - stub
        pass

    stub_mod.DualScaler = DualScaler
    stub_pkg.utils_scaler = stub_mod
    saved = {k: sys.modules.get(k) for k in ("src", "src.utils_scaler")}
    sys.modules["src"], sys.modules["src.utils_scaler"] = stub_pkg, stub_mod
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as f:
                s = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not unpickle scaler: {exc}"}
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    q_cols = cfg.get("features", {}).get("query_cols", [])
    m_cols = cfg.get("features", {}).get("metric_cols", [])
    out: dict[str, Any] = {}
    try:
        out["n_train_samples"] = int(s.q_scaler.n_samples_seen_)
        out["query_features"] = {c: {"mean": float(m), "std": float(sd)} for c, m, sd in zip(q_cols, s.q_scaler.mean_, s.q_scaler.scale_, strict=False)}
        out["metric_features"] = {c: {"mean": float(m), "std": float(sd)} for c, m, sd in zip(m_cols, s.m_scaler.mean_, s.m_scaler.scale_, strict=False)}
    except AttributeError as exc:
        out["error"] = str(exc)
    return out


# --------------------------------------------------------------------------- case database
def audit_case_database(db: CaseDatabase) -> dict[str, Any]:
    """Compute audit statistics for a built case database."""
    df = db.cases
    idc, mc, lc = db.id_col, db.mall_id_col, db.label_col
    num_cols = [c for c in db.query_cols + db.metric_cols + db.graph_metric_cols if c in df.columns]

    out: dict[str, Any] = {
        "source": db.manifest.get("source"),
        "synthetic": bool(db.manifest.get("synthetic", False)),
        "n_rows": int(len(df)),
        "n_malls": int(df[mc].nunique()),
        "n_graphs": int(len(db.graphs)),
        "columns": df.columns.tolist(),
    }
    out["missing_rate"] = {c: float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()}
    out["duplicates"] = {
        "duplicate_floor_ids": int(df[idc].duplicated().sum()),
        "duplicate_feature_rows": int(df[num_cols].round(6).duplicated().sum()) if num_cols else 0,
    }
    if lc in df:
        y = df[lc].dropna()
        out["label"] = {
            "count": int(len(y)),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "p05": float(y.quantile(0.05)),
            "median": float(y.median()),
            "p95": float(y.quantile(0.95)),
            "max": float(y.max()),
            "n_unique": int(y.nunique()),
            "skew": float(y.skew()),
        }
        # within-mall label variance vs. between-mall: tells whether the label is floor-level
        if len(df) > 1:
            grp = df.groupby(mc)[lc]
            within = float(grp.var().dropna().mean()) if grp.ngroups > 0 else float("nan")
            between = float(grp.mean().var()) if grp.ngroups > 1 else float("nan")
            out["label"]["within_mall_var"] = within
            out["label"]["between_mall_var"] = between
    if db.query_cols:
        # City-level features should be constant within a mall
        const = {}
        for c in db.query_cols:
            if c in df:
                const[c] = float((df.groupby(mc)[c].nunique(dropna=True) <= 1).mean())
        out["query_feature_constant_within_mall_rate"] = const
        # correlation of query features with label (spearman)
        if lc in df:
            out["spearman_with_label"] = {c: float(df[[c, lc]].corr(method="spearman").iloc[0, 1]) for c in num_cols if df[c].notna().sum() > 2}
    if "city_cluster" in df:
        out["city_cluster_counts"] = {str(k): int(v) for k, v in df["city_cluster"].value_counts().items()}
    if "layout_type" in df:
        out["layout_type_counts"] = {str(k): int(v) for k, v in df["layout_type"].value_counts(dropna=False).items()}
    if "split" in df:
        out["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
        tr, va, te = (set(df.loc[df.split == s, mc]) for s in ("train", "val", "test"))
        out["leakage"] = {"train_val_overlap_malls": len(tr & va), "train_test_overlap_malls": len(tr & te), "val_test_overlap_malls": len(va & te)}
    if db.graph_metric_cols and "g_num_nodes" in df:
        gn = df["g_num_nodes"].dropna()
        out["graph_size"] = {"min": int(gn.min()), "median": float(gn.median()), "max": int(gn.max())} if len(gn) else {}
        if "g_n_components" in df:
            out["disconnected_graph_rate"] = float((df["g_n_components"] > 1).mean())
    return out


# --------------------------------------------------------------------------- render
def render_markdown(title: str, report: dict[str, Any]) -> str:
    """Very small JSON→Markdown renderer (headings for top-level keys, fenced JSON below)."""
    lines = [f"# {title}", ""]
    for key, val in report.items():
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            lines.append("```")
            lines.extend(val)
            lines.append("```")
        else:
            lines.append("```json")
            lines.append(json.dumps(val, ensure_ascii=False, indent=2, default=str))
            lines.append("```")
        lines.append("")
    return "\n".join(lines)
