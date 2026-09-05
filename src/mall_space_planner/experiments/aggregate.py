"""Aggregate run.json files into tables (mean ± std over seeds) and figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_METRICS = ("ndcg@5", "ndcg@10", "precision@5", "recall@10", "hit@5", "map", "mrr", "spearman", "kendall_tau", "pairwise_acc")


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(root).rglob("run.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for split, m in (r.get("metrics") or {}).items():
            row = {"experiment": r.get("experiment_name"), "variant": r.get("variant", ""), "ranker": r.get("ranker"), "seed": r.get("seed"), "split": split, "n_queries": m.get("n_queries"), "fit_seconds": r.get("fit_seconds"), "run_dir": str(f.parent)}
            row.update({k: v for k, v in m.items() if not k.endswith("_std") and k not in ("n_queries", "split")})
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(root: Path, split: str = "test", metrics: tuple[str, ...] = DEFAULT_METRICS) -> pd.DataFrame:
    df = load_runs(root)
    if df.empty:
        return df
    df = df[df["split"] == split]
    keys = ["experiment", "variant", "ranker"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = df.groupby(keys)["seed"].nunique()
    agg["n_queries"] = df.groupby(keys)["n_queries"].mean()
    return agg.reset_index().sort_values(f"{metrics[0]}_mean", ascending=False)


def to_markdown_table(agg: pd.DataFrame, metrics: tuple[str, ...] = ("ndcg@5", "ndcg@10", "map", "spearman", "pairwise_acc"), decimals: int = 3) -> str:
    if agg.empty:
        return "_no runs found_"
    cols = [m for m in metrics if f"{m}_mean" in agg.columns]
    head = "| experiment | variant | ranker | seeds | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (4 + len(cols))
    lines = [head, sep]
    for _, r in agg.iterrows():
        cells = [f"{r[f'{m}_mean']:.{decimals}f} ± {0.0 if pd.isna(r[f'{m}_std']) else r[f'{m}_std']:.{decimals}f}" for m in cols]
        lines.append(f"| {r['experiment']} | {r['variant']} | {r['ranker']} | {int(r['n_seeds'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_metric_bars(agg: pd.DataFrame, metric: str, path: Path, title: str | None = None) -> Path | None:
    if agg.empty or f"{metric}_mean" not in agg.columns:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{e}\n{v}" if v else str(e) for e, v in zip(agg["experiment"], agg["variant"], strict=True)]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4))
    ax.bar(range(len(labels)), agg[f"{metric}_mean"], yerr=agg[f"{metric}_std"].fillna(0), capsize=4, color="#3a7bd5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} (mean ± std over seeds)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
