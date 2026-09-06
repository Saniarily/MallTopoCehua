"""Stage-2 result figures (R09–R14) from the round-4 multi-seed table and checkpoint training histories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mall_space_planner.reporting.style import color_for, fig, label, load_style, savefig, title_for

import matplotlib.pyplot as plt  # noqa: E402

MAIN = ["ref_ground_truth", "stage2_rule_baseline", "stage2_search_baseline", "stage2_ar_gnn", "stage2_ar_gnn_bestof16"]
ABL = ["stage2_ar_gnn", "stage2_ar_gnn_bfs_order", "stage2_ar_gnn_single_label", "stage2_ar_gnn_basic_feats", "stage2_ar_gnn_long"]


def _summary(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / "stage2/r4_summary_mean_std.csv").set_index("experiment")


def _per_seed(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / "stage2/r4_per_seed.csv")


def _short(name: str) -> str:
    return label("methods", name)


def r09_overview(results: Path, out: Path) -> list[Path]:
    """2x3 panel: pass rate, ASPL dev, density dev, attach precision, degree EMD, target-edge recall — main methods."""
    df = _summary(results)
    rows = [m for m in MAIN if m in df.index]
    metrics = ["overall_pass", "aspl_deviation_pct", "density_deviation_pct", "attach_precision_pct", "degree_emd", "target_edge_recall_pct"]
    better = ["↑", "↓", "↓", "↑", "↓", "↑"]
    s = load_style()
    f, axes = plt.subplots(2, 3, figsize=(s["figure"]["width_double"], 4.6))
    for ax, m, b in zip(axes.ravel(), metrics, better):
        vals = df.loc[rows, m + "_mean"].values
        errs = df.loc[rows, m + "_std"].values
        if m == "overall_pass":
            vals, errs = vals * 100, errs * 100
        cols = [color_for(r) for r in rows]
        ax.bar(range(len(rows)), vals, yerr=errs, color=cols, edgecolor="white", capsize=3, width=0.66, error_kw={"lw": 0.8})
        top = max(vals + errs)
        for i, (v, e) in enumerate(zip(vals, errs)):
            # place the value above the error-bar cap, never through it
            ax.text(i, v + e + top * 0.03, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=s["fonts"]["size_annot"])
        ax.set_title(("合格率 (%)" if m == "overall_pass" else label("metrics", m)) + f"  {b}", fontsize=s["fonts"]["size_label"])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(["真实", "规则", "规则\n+择优", "本文", "本文\n+择优"], fontsize=s["fonts"]["size_annot"])
        ax.grid(axis="x", alpha=0)
        ax.set_ylim(0, top * 1.25)
    f.suptitle(title_for("R09") + "（600 个留出骨架，3 次随机种子）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R09_stage2_overview")


def r10_attach_scatter(results: Path, out: Path) -> list[Path]:
    """Per-seed scatter: attach recall (x) vs attach precision (y); size = pass rate."""
    ps = _per_seed(results)
    df = _summary(results)
    s = load_style()
    f, ax = fig("double", 0.6)
    for name, g in ps.groupby("experiment"):
        if name == "ref_ground_truth":
            continue
        ax.scatter(g["attach_recall_pct"], g["attach_precision_pct"], s=18 + 140 * (g["overall_pass"] - 0.7).clip(0, 1), color=color_for(name), alpha=0.85, edgecolor="white", lw=0.5, label=_short(name))
    # Labels are placed in free space (data coords) with a thin leader line to the mean position, so
    # they never sit on top of each other or on top of points.
    ann = {
        "stage2_rule_baseline": (44.5, 39.0, "left"),
        "stage2_search_baseline": (46.5, 45.5, "left"),
        "stage2_ar_gnn_greedy": (42.0, 87.5, "left"),
        "stage2_ar_gnn": (52.0, 82.5, "right"),
        "stage2_ar_gnn_bestof16": (57.0, 68.5, "left"),
        "stage2_ar_gnn_long": (66.0, 82.5, "center"),
        "stage2_ar_gnn_bfs_order": (68.5, 79.5, "right"),
        "stage2_ar_gnn_basic_feats": (68.5, 69.0, "right"),
        "stage2_ar_gnn_single_label": (49.0, 72.5, "left"),
    }
    for name, (tx, ty, ha) in ann.items():
        if name not in df.index:
            continue
        txt = _short(name).split("（")[0].replace("自回归图网络", "本文").replace("消融：", "")
        ax.annotate(txt, (df.loc[name, "attach_recall_pct_mean"], df.loc[name, "attach_precision_pct_mean"]), xytext=(tx, ty), textcoords="data", fontsize=s["fonts"]["size_annot"], color="#333", ha=ha, va="center",
                    arrowprops={"arrowstyle": "-", "color": "#999", "lw": 0.6, "shrinkB": 4})
    ax.set_xlabel(label("metrics", "attach_recall_pct") + "  — 真实分支位置被找到的比例")
    ax.set_ylabel(label("metrics", "attach_precision_pct") + "  — 生成分支位置正确的比例")
    ax.set_title(title_for("R10") + "（点越大 = 大纲合格率越高）", loc="left", fontweight="bold")
    ax.set_xlim(30, 72)
    ax.set_ylim(35, 92)
    ax.text(0.98, 0.03, "每个点 = 一次随机种子的评估；标注 = 各方法的均值位置", transform=ax.transAxes, ha="right", va="bottom", fontsize=s["fonts"]["size_annot"], color="#666")
    return savefig(f, out, "R10_stage2_attach_recall_precision")


def r11_decoding_tradeoff(results: Path, out: Path) -> list[Path]:
    """Greedy / sampling / best-of-16: pass rate & ASPL dev vs attach precision."""
    df = _summary(results)
    rows = [r for r in ["stage2_search_baseline", "stage2_ar_gnn_greedy", "stage2_ar_gnn", "stage2_ar_gnn_bestof16", "stage2_ar_gnn_long_bestof16"] if r in df.index]
    s = load_style()
    f, ax = fig("double", 0.58)
    ax2 = ax.twinx()
    x = np.arange(len(rows))
    w = 0.36
    b1 = ax.bar(x - w / 2, df.loc[rows, "overall_pass_mean"] * 100, w, yerr=df.loc[rows, "overall_pass_std"] * 100, color=s["palette"]["ours"], edgecolor="white", capsize=3, label="大纲合格率 (%)")
    b2 = ax2.bar(x + w / 2, df.loc[rows, "aspl_deviation_pct_mean"], w, yerr=df.loc[rows, "aspl_deviation_pct_std"], color=s["palette"]["main"][1], edgecolor="white", capsize=3, label="平均步行路径偏差 (%)")
    ax.plot(x, df.loc[rows, "attach_precision_pct_mean"], "D-", color=s["palette"]["highlight"], ms=6, lw=1.4, label="分支位置正确率 (%)")
    ax.set_ylim(0, 128)  # head-room for the legend above the bars
    ax2.set_ylim(0, 48)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(r).replace("自回归图网络（本文）", "本文").replace("自回归图网络", "本文").replace("（本文推荐）", "") for r in rows], rotation=12, ha="right")
    ax.set_ylabel("合格率 / 分支正确率 (%)")
    ax2.set_ylabel("平均步行路径偏差 (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=s["fonts"]["size_annot"])
    ax.set_title(title_for("R11") + "：结构正确 vs 走廊长度合理", loc="left", fontweight="bold")
    return savefig(f, out, "R11_stage2_decoding_tradeoff")


def r12_ablation(results: Path, out: Path) -> list[Path]:
    df = _summary(results)
    rows = [r for r in ABL if r in df.index]
    metrics = [("attach_precision_pct", "↑"), ("target_edge_recall_pct", "↑"), ("degree_emd", "↓")]
    s = load_style()
    f, axes = plt.subplots(1, 3, figsize=(s["figure"]["width_double"], 2.9), sharey=True)
    y = np.arange(len(rows))
    for ax, (m, b) in zip(axes, metrics):
        vals = df.loc[rows, m + "_mean"].values
        errs = df.loc[rows, m + "_std"].values
        cols = [s["palette"]["highlight"] if r == "stage2_ar_gnn" else (s["palette"]["main"][2] if "long" in r else s["palette"]["baseline"]) for r in rows]
        ax.barh(y, vals, xerr=errs, color=cols, edgecolor="white", capsize=3, height=0.62)
        for yy, v, e in zip(y, vals, errs):
            ax.text(v + e + (0.01 if m == "degree_emd" else 0.6), yy, f"{v:.2f}" if m == "degree_emd" else f"{v:.1f}", va="center", fontsize=s["fonts"]["size_annot"])
        ax.set_title(label("metrics", m).replace(" (", "\n(") + f" {b}", fontsize=s["fonts"]["size_label"], linespacing=1.15)
        ax.grid(axis="y", alpha=0)
        lo = min(vals - errs)
        hi = max(vals + errs)
        ax.set_xlim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_short(r).replace("自回归图网络（本文）", "完整模型（本文）") for r in rows])
    axes[0].invert_yaxis()
    f.suptitle(title_for("R12") + "（去掉某一成分后的变化）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R12_stage2_ar_gnn_ablation")


def r13_training_curves(results: Path, out: Path) -> list[Path]:
    ck = results / "stage2/checkpoints"
    runs = [("ar_gnn_v1", "第一版（失败对照）"), ("ar_gnn", "本文模型"), ("ar_gnn_long", "大模型 20 轮"), ("ar_gnn_bfs_order", "消融：广度优先顺序"), ("ar_gnn_single_label", "消融：单一接点"), ("ar_gnn_basic_feats", "消融：去结构特征")]
    s = load_style()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(s["figure"]["width_double"], 3.0))
    for i, (name, lab) in enumerate(runs):
        p = ck / name / "meta.json"
        if not p.exists():
            continue
        h = json.load(open(p, encoding="utf-8"))["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ls = "--" if name == "ar_gnn_v1" else "-"
        c = s["palette"]["reference"] if name == "ar_gnn_v1" else (s["palette"]["highlight"] if name == "ar_gnn" else s["palette"]["main"][i % 8])
        a1.plot(ep, h["train_loss"], ls, color=c, lw=1.4, label=lab)
        a2.plot(ep, np.array(h["val_anchor_acc"]) * 100, ls, color=c, lw=1.4, label=lab)
    a1.set_xlabel("训练轮次")
    a1.set_ylabel(label("metrics", "train_loss"))
    a1.set_title("(a) 训练损失", loc="left", fontsize=s["fonts"]["size_label"])
    a2.set_xlabel("训练轮次")
    a2.set_ylabel("接点预测准确率 (%)")
    a2.set_title("(b) 验证集：下一节点接到哪里 预测正确率", loc="left", fontsize=s["fonts"]["size_label"])
    a2.axhline(5.4, color="#999", lw=0.8, ls=":")
    a2.text(1, 6.2, "随机猜测 ≈ 5%", fontsize=s["fonts"]["size_annot"], color="#777")
    a2.legend(fontsize=s["fonts"]["size_annot"], loc="center right")
    f.suptitle(title_for("R13"), x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R13_stage2_training_curves")


def r14_growth_pattern(results: Path, out: Path) -> list[Path]:
    """new-new ratio (corridor-like growth) and degree EMD vs ground truth."""
    df = _summary(results)
    rows = [m for m in ["stage2_rule_baseline", "stage2_search_baseline", "stage2_ar_gnn_greedy", "stage2_ar_gnn", "stage2_ar_gnn_bestof16"] if m in df.index]
    gt = df.loc["ref_ground_truth", "new_new_ratio_gen_mean"]
    s = load_style()
    f, (a1, a2) = plt.subplots(1, 2, figsize=(s["figure"]["width_double"], 2.9))
    x = np.arange(len(rows))
    a1.bar(x, df.loc[rows, "new_new_ratio_gen_mean"], yerr=df.loc[rows, "new_new_ratio_gen_std"], color=[color_for(r) for r in rows], edgecolor="white", capsize=3, width=0.64)
    a1.axhline(gt, color=s["palette"]["ground_truth"], ls="--", lw=1.2)
    a1.text(-0.4, gt + 0.015, f"真实建成拓扑 {gt:.2f}", ha="left", fontsize=s["fonts"]["size_annot"], color=s["palette"]["ground_truth"])
    a1.set_ylim(0, 0.62)
    a1.set_ylabel("新节点之间连边的比例")
    a1.set_title("(a) 走廊式生长 vs 全部挂回骨架", loc="left", fontsize=s["fonts"]["size_label"])
    a2.bar(x, df.loc[rows, "degree_emd_mean"], yerr=df.loc[rows, "degree_emd_std"], color=[color_for(r) for r in rows], edgecolor="white", capsize=3, width=0.64)
    a2.set_ylabel(label("metrics", "degree_emd"))
    a2.set_title("(b) 节点连接度分布与真实的差异", loc="left", fontsize=s["fonts"]["size_label"])
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(["规则", "规则\n+择优", "本文\n贪心", "本文", "本文\n+择优"], fontsize=s["fonts"]["size_annot"])
        ax.grid(axis="x", alpha=0)
    f.suptitle(title_for("R14"), x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R14_stage2_growth_pattern")


ALL = [r09_overview, r10_attach_scatter, r11_decoding_tradeoff, r12_ablation, r13_training_curves, r14_growth_pattern]
