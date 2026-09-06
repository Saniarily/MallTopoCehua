"""Stage-1 result figures (R01–R08). Each function: results_dir -> list of written files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mall_space_planner.reporting.style import color_for, fig, label, load_style, parse_pm, read_md_table, savefig, title_for

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _model_table(results: Path) -> pd.DataFrame:
    df = read_md_table(results / "stage1/model_comparison_table_test.md")
    for m in ("ndcg@5", "ndcg@10", "map", "spearman", "pairwise_acc"):
        df[[m, m + "_std"]] = df[m].apply(lambda c: pd.Series(parse_pm(c)))
    df["key"] = df["experiment"].str.replace("stage1_", "").str.replace("ref_", "")
    df["key"] = df["key"].replace({"quality_oracle_upper_bound": "quality_oracle", "random_lower_bound": "random", "rule_knn": "weighted_rule"})
    return df


def r01_model_comparison(results: Path, out: Path) -> list[Path]:
    """Horizontal bars of NDCG@10 with seed error bars; oracle / random as dashed reference lines."""
    df = _model_table(results).sort_values("ndcg@10")
    ref = df[df["key"].isin(["quality_oracle", "random"])]
    body = df[~df["key"].isin(["quality_oracle", "random"])]
    f, ax = fig("double", 0.62)
    y = np.arange(len(body))
    cols = [color_for(k) for k in body["key"]]
    ax.barh(y, body["ndcg@10"], xerr=body["ndcg@10_std"], color=cols, edgecolor="white", capsize=3, height=0.66, error_kw={"lw": 0.9})
    ax.set_yticks(y)
    ax.set_yticklabels([label("methods", k) for k in body["key"]])
    for yy, v in zip(y, body["ndcg@10"]):
        ax.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=load_style()["fonts"]["size_annot"])
    for _, r in ref.iterrows():
        ax.axvline(r["ndcg@10"], ls="--", lw=1, color=load_style()["palette"]["reference"])
        ax.text(r["ndcg@10"] + (0.004 if r["key"] == "random" else -0.004), len(body) - 0.55, f"{label('methods', r['key'])}\n{r['ndcg@10']:.3f}", fontsize=load_style()["fonts"]["size_annot"], color="#555", ha="left" if r["key"] == "random" else "right", va="center")
    ax.set_ylim(-0.6, len(body) + 0.2)
    ax.set_xlim(0.45, 0.80)
    ax.set_xlabel(label("metrics", "ndcg@10") + "  — 越高越好，误差棒 = 3 次随机种子的标准差")
    ax.set_title(title_for("R01"), loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0)
    return savefig(f, out, "R01_stage1_model_comparison")


def r02_multi_metric(results: Path, out: Path) -> list[Path]:
    """Grouped bars: NDCG@5 / NDCG@10 / MAP / PairAcc for the main methods."""
    df = _model_table(results)
    keep = ["quality_oracle", "extra_trees", "deep_residual", "ridge", "lgbm_regressor", "lgbm_lambdarank", "random_forest", "mlp", "random"]
    df = df.set_index("key").loc[[k for k in keep if k in set(df["key"])]].reset_index()
    metrics = ["ndcg@5", "ndcg@10", "map", "pairwise_acc"]
    f, ax = fig("double", 0.55)
    w = 0.8 / len(metrics)
    x = np.arange(len(df))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * w, df[m], w, yerr=df[m + "_std"], label=label("metrics", m), color=load_style()["palette"]["main"][i], edgecolor="white", capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels([label("methods", k).split("（")[0] for k in df["key"]], rotation=20, ha="right")
    ax.set_ylim(0.15, 1.0)
    ax.set_ylabel("指标值（越高越好）")
    ax.legend(ncol=2, loc="upper right")
    ax.set_title(title_for("R02"), loc="left", fontweight="bold")
    return savefig(f, out, "R02_stage1_multi_metric")


def _ablation_table(path: Path, metric_cols=("ndcg@10", "spearman", "pairwise_acc")) -> pd.DataFrame:
    df = read_md_table(path)
    for m in metric_cols:
        df[[m, m + "_std"]] = df[m].apply(lambda c: pd.Series(parse_pm(c)))
    return df


def r03_feature_blocks(results: Path, out: Path) -> list[Path]:
    """Feature-block ablation: change vs full for NDCG@10 / Spearman / PairAcc (dot-and-whisker)."""
    df = _ablation_table(results / "stage1/feature_blocks_table_test.md")
    full = df[df["variant"] == "full"].iloc[0]
    df = df[df["variant"] != "full"]
    metrics = ["ndcg@10", "spearman", "pairwise_acc"]
    f, axes = plt.subplots(1, 3, figsize=(load_style()["figure"]["width_double"], 2.9), sharey=True)
    y = np.arange(len(df))
    for ax, m, c in zip(axes, metrics, load_style()["palette"]["main"]):
        delta = df[m] - full[m]
        ax.errorbar(delta, y, xerr=df[m + "_std"], fmt="o", color=c, capsize=3, lw=1)
        ax.axvline(0, color="#444", lw=1)
        ax.axvspan(-full[m + "_std"], full[m + "_std"], color="#999", alpha=0.15, lw=0)
        ax.set_title(label("metrics", m), fontsize=load_style()["fonts"]["size_label"])
        ax.set_xlabel("相对完整特征的变化")
        ax.grid(axis="y", alpha=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label("variants", v) for v in df["variant"]])
    f.suptitle(title_for("R03") + "（灰带 = 完整特征的种子波动范围）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R03_stage1_feature_block_ablation")


def r04_deep_ablation(results: Path, out: Path) -> list[Path]:
    df = _ablation_table(results / "stage1/deep_ablation_table_test.md")
    order = ["full_residual_tf_gnn", "no_gnn", "mlp_instead_of_transformer", "no_smallsample_tricks", "no_residual_end2end"]
    df = df.set_index("variant").loc[[o for o in order if o in df["variant"].values]].reset_index()
    base = _model_table(results).set_index("key")
    et = base.loc["extra_trees", "ndcg@10"]
    mlp = base.loc["mlp", "ndcg@10"]
    f, ax = fig("double", 0.55)
    y = np.arange(len(df))
    cols = [load_style()["palette"]["highlight"] if v == "full_residual_tf_gnn" else load_style()["palette"]["main"][i % 8] for i, v in enumerate(df["variant"])]
    ax.barh(y, df["ndcg@10"], xerr=df["ndcg@10_std"], color=cols, edgecolor="white", capsize=3, height=0.62)
    for yy, v, s in zip(y, df["ndcg@10"], df["ndcg@10_std"]):
        ax.text(v + s + 0.004, yy, f"{v:.3f} ± {s:.3f}", va="center", fontsize=load_style()["fonts"]["size_annot"])
    ax.axvline(et, ls="--", color=color_for("extra_trees"), lw=1.2, label=f"极端随机树（最强经典模型）{et:.3f}")
    ax.axvline(mlp, ls=":", color=color_for("mlp"), lw=1.2, label=f"纯表格 MLP（旧方案）{mlp:.3f}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=load_style()["fonts"]["size_annot"])
    ax.set_yticks(y)
    ax.set_yticklabels([label("variants", v) for v in df["variant"]])
    ax.invert_yaxis()
    ax.set_xlim(0.55, 0.76)
    ax.set_xlabel(label("metrics", "ndcg@10"))
    ax.set_title(title_for("R04"), loc="left", fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0)
    return savefig(f, out, "R04_stage1_deep_ablation")


def _tables_from_summary_md(path: Path) -> dict:
    """Fallback: parse the per-cluster tables written by evaluate_type_recommender.py (summary.md)."""
    if not path.exists():
        return {}
    import re
    tables, cur = {}, None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## cluster (\S+)", ln)
        if m:
            cur = m.group(1); tables[cur] = []; continue
        if cur and ln.startswith("|") and not ln.startswith("| rank") and not ln.startswith("|---"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            ci = re.findall(r"[-0-9.]+", c[3])
            tables[cur].append({"rank": int(c[0]), "layout_type": c[1], "expected_score": float(c[2]), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "empirical_mean": None if c[4] == "-" else float(c[4]), "empirical_n": int(c[5])})
    return tables


def r05_type_tables(results: Path, out: Path) -> list[Path]:
    """Per-cluster expected score by layout type with bootstrap CI, empirical mean as hollow markers, n as text."""
    res = json.load(open(results / "stage1/type_recommender_results.json", encoding="utf-8"))
    last = res[-1]
    tables = last.get("type_tables") or _tables_from_summary_md(results / "stage1/type_recommender_summary.md")
    if not tables:
        return []
    clusters = sorted(tables, key=int)
    s = load_style()
    f, axes = plt.subplots(1, len(clusters), figsize=(s["figure"]["width_double"], 3.0), sharex=False)
    for ax, c in zip(np.atleast_1d(axes), clusters):
        rows = tables[c]
        y = np.arange(len(rows))
        exp = np.array([r["expected_score"] for r in rows])
        lo = np.array([r["ci_low"] for r in rows])
        hi = np.array([r["ci_high"] for r in rows])
        cols = [s["palette"]["highlight"] if r["rank"] == 1 else s["palette"]["ours"] for r in rows]
        ax.hlines(y, lo, hi, color=cols, lw=2.2)
        ax.scatter(exp, y, color=cols, s=28, zorder=3, label="模型期望评分（线段 = 80% 置信区间）")
        emp = [r["empirical_mean"] for r in rows]
        ax.scatter([e if e is not None else np.nan for e in emp], y, facecolors="none", edgecolors="#333", s=34, zorder=3, label="可比案例实际平均评分")
        xr = max(hi) + 0.16
        for yy, r in zip(y, rows):
            ax.text(xr, yy, f"n={r['empirical_n']}", va="center", ha="right", fontsize=s["fonts"]["size_annot"], color="#555")
        ax.set_yticks(y)
        ax.set_yticklabels([r["layout_type"] for r in rows])
        ax.invert_yaxis()
        ax.set_title(label("clusters", c), fontsize=s["fonts"]["size_label"])
        ax.set_xlabel("期望综合评分")
        ax.grid(axis="y", alpha=0)
        ax.set_xlim(min(lo) - 0.04, max(hi) + 0.17)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    f.legend(h[:2], l[:2], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    f.suptitle(title_for("R05") + "（红色 = 该类城市下预期评分最高的类型）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R05_stage1_type_expected_score_by_cluster")


def r06_type_model_value(results: Path, out: Path) -> list[Path]:
    """(a) with-type vs conditions-only RMSE/Spearman across seeds; (b) policy uplift per cluster."""
    res = json.load(open(results / "stage1/type_recommender_results.json", encoding="utf-8"))
    s = load_style()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(s["figure"]["width_double"], 2.9))
    seeds = [r["seed"] for r in res]
    x = np.arange(len(seeds))
    a1.plot(x, [r["spearman_with_type"] for r in res], "o-", color=s["palette"]["highlight"], label="带布局类型")
    a1.plot(x, [r["spearman_conditions_only"] for r in res], "s--", color=s["palette"]["reference"], label="仅策划条件")
    a1.set_xticks(x)
    a1.set_xticklabels([f"种子{k}" for k in seeds])
    a1.set_ylabel("预测评分与真实评分的排序相关性")
    a1.set_title("(a) 加入布局类型后的解释力", loc="left", fontsize=s["fonts"]["size_label"])
    a1.legend()
    # (b)
    per = {}
    for r in res:
        for c in r["per_cluster"]:
            per.setdefault(c["cluster"], []).append(c["score_when_type_matches_rec"] - c["score_when_type_differs"])
    cl = sorted(per)
    means = [np.mean(per[c]) for c in cl]
    stds = [np.std(per[c]) for c in cl]
    cols = [s["palette"]["ours"] if m >= 0 else s["palette"]["baseline"] for m in means]
    a2.bar(range(len(cl)), means, yerr=stds, color=cols, capsize=3, edgecolor="white")
    a2.axhline(0, color="#444", lw=1)
    a2.set_xticks(range(len(cl)))
    a2.set_xticklabels([label("clusters", str(c)).replace("城市", "\n城市") for c in cl])
    a2.set_ylabel("评分差（按推荐类型建 − 未按推荐建）")
    a2.set_title("(b) 采纳推荐类型的评分收益", loc="left", fontsize=s["fonts"]["size_label"])
    for i, (m, sd) in enumerate(zip(means, stds)):
        a2.text(i, m + (sd + 0.01 if m >= 0 else -sd - 0.03), f"{m:+.2f}", ha="center", fontsize=s["fonts"]["size_annot"])
    f.suptitle(title_for("R06"), x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R06_stage1_type_model_value")


def r07_fidelity(results: Path, out: Path) -> list[Path]:
    df = pd.read_csv(results / "stage1/fidelity_summary.csv")
    order = ["ref_oracle", "ref_majority", "ref_random", "stage1_rule_knn", "stage1_ridge", "stage1_lgbm_lambdarank", "stage1_extra_trees", "stage1_deep_residual", "ref_quality_oracle_upper_bound"]
    names = set(df["name"])
    df = df.set_index("name").loc[[o for o in order if o in names]].reset_index()
    s = load_style()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(s["figure"]["width_double"], 3.2), sharey=True)
    y = np.arange(len(df))
    cols = [s["palette"]["reference"] if n.startswith("ref_") and "quality" not in n else (s["palette"]["highlight"] if "deep" in n else s["palette"]["ours"]) for n in df["name"]]
    a1.barh(y, df["type_precision@5"], color=cols, edgecolor="white", height=0.62)
    a1.set_xlabel(label("metrics", "type_precision@5"))
    a1.set_xlim(0.4, 1.05)
    a2.barh(y, df["quality@5"], color=cols, edgecolor="white", height=0.62)
    a2.set_xlabel(label("metrics", "quality@5"))
    a2.set_xlim(4.0, 5.0)
    for ax, col in ((a1, "type_precision@5"), (a2, "quality@5")):
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for yy, v in zip(y, df[col]):
            ax.text(v + 0.012 * span, yy, f"{v:.2f}", va="center", fontsize=s["fonts"]["size_annot"])
        ax.grid(axis="y", alpha=0)
    a1.set_yticks(y)
    a1.set_yticklabels([label("methods", n) for n in df["name"]])
    a1.invert_yaxis()
    f.suptitle(title_for("R07") + "：左 = 类型是否一致，右 = 推荐案例的实际评分", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R07_stage1_prototype_fidelity")


def r08_layout_predictability(results: Path, out: Path) -> list[Path]:
    d = json.load(open(results / "stage1/layout_predictability.json", encoding="utf-8"))
    s = load_style()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(s["figure"]["width_double"], 2.8), gridspec_kw={"width_ratios": [1, 1.6]})
    a1.bar([0, 1], [d["majority_accuracy"], d["lgbm_accuracy"]], color=[s["palette"]["reference"], s["palette"]["ours"]], edgecolor="white", width=0.6)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["永远猜\n最常见类型", "由策划条件\n预测类型"])
    a1.set_ylim(0, 0.5)
    a1.set_ylabel("预测正确率")
    for i, v in enumerate([d["majority_accuracy"], d["lgbm_accuracy"]]):
        a1.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=s["fonts"]["size_annot"])
    a1.set_title("(a) 条件能否预测业主实际选了哪种类型", loc="left", fontsize=s["fonts"]["size_label"])
    imp = pd.Series(d["lgbm_feature_importance"]).sort_values()
    a2.barh(range(len(imp)), imp.values / imp.values.sum(), color=s["palette"]["ours"], edgecolor="white")
    a2.set_yticks(range(len(imp)))
    a2.set_yticklabels([label("conditions", k) for k in imp.index])
    a2.set_xlabel("相对重要性")
    a2.set_title("(b) 预测时依赖的条件", loc="left", fontsize=s["fonts"]["size_label"])
    a2.grid(axis="y", alpha=0)
    f.suptitle(title_for("R08"), x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R08_layout_type_predictability")


ALL = [r01_model_comparison, r02_multi_metric, r03_feature_blocks, r04_deep_ablation, r05_type_tables, r06_type_model_value, r07_fidelity, r08_layout_predictability]
