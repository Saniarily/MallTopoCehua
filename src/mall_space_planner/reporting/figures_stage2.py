"""Stage-2 result figures (R09–R14) from the round-5 (real corpus v2) multi-seed table and checkpoint training
histories. Falls back to the round-4 (superseded, LLM-corpus) snapshot only if r5 files are absent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mall_space_planner.reporting.style import color_for, fig, label, load_style, savefig, title_for

import matplotlib.pyplot as plt  # noqa: E402

MAIN = ["ref_ground_truth", "stage2_rule_baseline", "stage2_search_baseline", "stage2_ar_gnn", "stage2_ar_gnn_bestof16"]
# v3 ablations (round 5): growth order (bfs / greedy), single-anchor loss, planarity guard off; legacy names kept for r4 fallback
ABL = ["stage2_ar_gnn", "stage2_ar_gnn_bfs_order", "stage2_ar_gnn_greedy_order", "stage2_ar_gnn_single_label", "stage2_ar_gnn_no_planar", "stage2_ar_gnn_basic_feats", "stage2_ar_gnn_long"]


def _round(results: Path) -> str:
    return "r5" if (results / "stage2/r5_summary_mean_std.csv").exists() else "r4"


def _summary(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / f"stage2/{_round(results)}_summary_mean_std.csv").set_index("experiment")


def _per_seed(results: Path) -> pd.DataFrame | None:
    p = results / f"stage2/{_round(results)}_per_seed.csv"
    return pd.read_csv(p) if p.exists() else None


def _n_eval(results: Path) -> str:
    return "575 个测试骨架（按商场分组的真实语料）" if _round(results) == "r5" else "600 个留出骨架"


def _short(name: str) -> str:
    return label("methods", name)


def r09_overview(results: Path, out: Path) -> list[Path]:
    """2x3 panel: pass rate, ASPL dev, density dev, attach precision, degree EMD, target-edge recall — main methods."""
    df = _summary(results)
    rows = [m for m in MAIN if m in df.index]
    # On the real corpus target_edge_recall is identical for every generator (= skeleton edges / target edges, the
    # skeleton being kept verbatim), so the 6th panel shows attach *recall* instead (r4 fallback keeps the old panel).
    last = "attach_recall_pct" if _round(results) == "r5" else "target_edge_recall_pct"
    metrics = ["overall_pass", "aspl_deviation_pct", "density_deviation_pct", "attach_precision_pct", "degree_emd", last]
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
    f.suptitle(title_for("R09") + f"（{_n_eval(results)}，3 次随机种子）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "R09_stage2_overview")


def r10_attach_scatter(results: Path, out: Path) -> list[Path]:
    """Attach recall (x) vs attach precision (y). Marker = mean over seeds (±1 std error bars), faint dots = single
    seeds. Methods are identified by a number printed at the marker and a numbered key at the right (no leader lines);
    the dense AR-GNN cluster is repeated in a zoom inset so the numbers stay readable."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    ps = _per_seed(results)
    df = _summary(results)
    s = load_style()
    f = plt.figure(figsize=(s["figure"]["width_double"] * 1.15, s["figure"]["width_double"] * 0.60))
    gs = f.add_gridspec(1, 2, width_ratios=[1.0, 0.40], wspace=0.02, left=0.08, right=0.995, top=0.90, bottom=0.12)
    ax = f.add_subplot(gs[0, 0]); key = f.add_subplot(gs[0, 1]); key.axis("off")
    order = ["stage2_rule_baseline", "stage2_search_baseline", "stage2_ar_gnn", "stage2_ar_gnn_bestof16", "stage2_ar_gnn_greedy", "stage2_ar_gnn_bfs_order", "stage2_ar_gnn_greedy_order", "stage2_ar_gnn_single_label", "stage2_ar_gnn_no_planar", "stage2_ar_gnn_long", "stage2_ar_gnn_long_bestof16", "stage2_ar_gnn_basic_feats"]
    shown = [n for n in order if n in df.index] + [n for n in df.index if n not in order and n != "ref_ground_truth"]
    xs = df.loc[shown, "attach_recall_pct_mean"]; ys = df.loc[shown, "attach_precision_pct_mean"]

    def _draw(a, lab: bool, fs: float, ms_scale: float = 1.0) -> None:  # noqa: ANN001
        for i, name in enumerate(shown, start=1):
            c = color_for(name)
            ours = name in ("stage2_ar_gnn", "stage2_ar_gnn_bestof16")
            x, y = float(df.loc[name, "attach_recall_pct_mean"]), float(df.loc[name, "attach_precision_pct_mean"])
            if ps is not None:
                g = ps[ps["experiment"] == name]
                a.scatter(g["attach_recall_pct"], g["attach_precision_pct"], s=14 * ms_scale, color=c, alpha=0.35, edgecolor="none", zorder=2)
            a.errorbar(x, y, xerr=df.loc[name, "attach_recall_pct_std"], yerr=df.loc[name, "attach_precision_pct_std"], fmt="D" if ours else "o", ms=(10 if ours else 8.5) * ms_scale, color=c, ecolor=c, elinewidth=0.9, capsize=2.5, mec="white", mew=0.8, zorder=4)
            if lab:
                a.text(x, y, str(i), ha="center", va="center", fontsize=fs, color="white", fontweight="bold", zorder=5)

    _draw(ax, True, s["fonts"]["size_annot"] - 1.5)
    pad = 4.0
    # extra room at the bottom-left for the zoom inset (the rule methods sit above it)
    ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad); ax.set_ylim(float(ys.min()) - pad - 14, float(ys.max()) + pad)
    ax.set_xlabel(label("metrics", "attach_recall_pct") + "  — 真实分支位置被找到的比例")
    ax.set_ylabel(label("metrics", "attach_precision_pct") + "  — 生成分支位置正确的比例")
    ax.grid(alpha=s["figure"]["grid_alpha"])
    # zoom inset on the densest cluster (the AR-GNN family around the full model)
    cx, cy = float(df.loc["stage2_ar_gnn", "attach_recall_pct_mean"]), float(df.loc["stage2_ar_gnn", "attach_precision_pct_mean"])
    near = [n for n in shown if abs(float(df.loc[n, "attach_recall_pct_mean"]) - cx) < 7 and abs(float(df.loc[n, "attach_precision_pct_mean"]) - cy) < 4]
    if len(near) >= 3:
        nx_ = df.loc[near, "attach_recall_pct_mean"]; ny_ = df.loc[near, "attach_precision_pct_mean"]
        x0, x1, y0, y1 = float(nx_.min()) - 2.0, float(nx_.max()) + 2.0, float(ny_.min()) - 1.6, float(ny_.max()) + 1.6
        # inset in the empty lower-left quadrant (rule methods sit at low recall / low precision but far left)
        axin = inset_axes(ax, width="48%", height="42%", loc="lower right", borderpad=1.4)
        _draw(axin, True, s["fonts"]["size_annot"] - 1.0, ms_scale=1.25)
        axin.set_xlim(x0, x1); axin.set_ylim(y0, y1)
        axin.tick_params(labelsize=s["fonts"]["size_annot"] - 2)
        axin.set_title("放大：AR-GNN 系列", fontsize=s["fonts"]["size_annot"] - 1, pad=2)
        axin.grid(alpha=s["figure"]["grid_alpha"])
        for sp in axin.spines.values():
            sp.set_edgecolor("#888")
        mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="#888", lw=0.7, ls="--")
    # numbered key
    key.set_xlim(0, 1); key.set_ylim(0, 1)
    yy = 0.99
    key.text(0.0, yy, "编号 → 方法（召回 / 正确率 %）", fontsize=s["fonts"]["size_annot"] + 0.5, fontweight="bold", va="top"); yy -= 0.085
    step = 0.9 / max(len(shown), 1)
    for i, name in enumerate(shown, start=1):
        c = color_for(name)
        ours = name in ("stage2_ar_gnn", "stage2_ar_gnn_bestof16")
        key.scatter([0.05], [yy - 0.012], s=110, marker="D" if ours else "o", color=c, edgecolor="white", linewidth=0.8)
        key.text(0.05, yy - 0.012, str(i), ha="center", va="center", fontsize=s["fonts"]["size_annot"] - 1.5, color="white", fontweight="bold")
        key.text(0.14, yy - 0.012, _short(name) + f"\n{float(df.loc[name, 'attach_recall_pct_mean']):.1f} / {float(df.loc[name, 'attach_precision_pct_mean']):.1f}", va="center", fontsize=s["fonts"]["size_annot"] - 0.5, linespacing=1.15, color="#222")
        yy -= step
    f.suptitle(title_for("R10") + "\n标记 = 3 个随机种子的均值，误差线 = ±1 标准差，浅点 = 单个种子；菱形 = 本文方法", x=0.08, ha="left", fontweight="bold", fontsize=s["fonts"]["size_title"] - 1, linespacing=1.3)
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
    ax2.set_zorder(ax.get_zorder() - 1); ax.patch.set_visible(False)  # bars behind, the line in front of *both* bar groups
    yv = df.loc[rows, "attach_precision_pct_mean"].to_numpy()
    ax.plot(x, yv, "-", color="white", lw=4.0, zorder=5, solid_capstyle="round")  # halo so the line reads over the orange bars
    ax.plot(x, yv, "D-", color=s["palette"]["highlight"], ms=6.5, lw=1.6, mec="white", mew=0.8, zorder=6, label="分支位置正确率 (%)")
    for xi, v in zip(x, yv):
        ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=s["fonts"]["size_annot"], color=s["palette"]["highlight"], zorder=7)
    ax.set_ylim(0, 128)  # head-room for the legend above the bars
    ax2.set_ylim(0, max(48.0, float((df.loc[rows, "aspl_deviation_pct_mean"] + df.loc[rows, "aspl_deviation_pct_std"]).max()) * 1.3))
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
    mid = "attach_recall_pct" if _round(results) == "r5" else "target_edge_recall_pct"
    metrics = [("attach_precision_pct", "↑"), (mid, "↑"), ("degree_emd", "↓")]
    s = load_style()
    f, axes = plt.subplots(1, 3, figsize=(s["figure"]["width_double"], 2.9), sharey=True)
    y = np.arange(len(rows))
    for ax, (m, b) in zip(axes, metrics):
        vals = df.loc[rows, m + "_mean"].values
        errs = df.loc[rows, m + "_std"].values
        cols = [s["palette"]["highlight"] if r == "stage2_ar_gnn" else (s["palette"]["main"][2] if ("long" in r or "no_planar" in r) else s["palette"]["baseline"]) for r in rows]
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
    if (results / "stage2/checkpoints_r5").exists():
        ck = results / "stage2/checkpoints_r5"
        runs = [("ar_gnn", "本文模型（语料生长顺序）"), ("ar_gnn_greedy_order", "消融：贪心生长顺序"), ("ar_gnn_bfs_order", "消融：广度优先生长顺序"), ("ar_gnn_single_label", "消融：只认单一接点")]
    else:
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
    # chance level = 1 / mean number of present nodes at a step (real corpus: ~30 present nodes -> ~3%)
    chance = 3.3 if _round(results) == "r5" else 5.4
    a2.axhline(chance, color="#999", lw=0.8, ls=":")
    a2.text(1, chance + 0.8, f"随机猜测 ≈ {chance:.0f}%", fontsize=s["fonts"]["size_annot"], color="#777")
    a2.set_ylim(0, None)
    a2.legend(fontsize=s["fonts"]["size_annot"], loc="lower right", bbox_to_anchor=(1.0, 0.12), framealpha=0.9)
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
    a1.set_ylim(0, max(0.62, float((df.loc[rows, "new_new_ratio_gen_mean"] + df.loc[rows, "new_new_ratio_gen_std"]).max()) * 1.35))
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


# --------------------------------------------------------------------------- stage 3 (renovation batch)
def r15_renovation_batch(results: Path, out: Path) -> list[Path]:
    """Renovation workflow, batch result on the filtered low-score test floors: paired before / after distributions of
    the circulation indicators (one panel per indicator; grey segments = individual floors, dots = means) and the share
    of floors improved, for every available version (v1 / v2 / v2b). Inputs: ``per_floor.csv`` written by
    ``scripts/renovate_stage3.py`` – looked up in ``<results>/stage3/renovation*/`` then ``outputs/experiments/renovation*/``.
    Returns [] when no batch has been run (the thesis marks R15 as pending until then)."""
    from mall_space_planner.stage3.renovate import BETTER, TOPO_LABEL

    cands = sorted([*results.glob("stage3/renovation*"), *Path("outputs/experiments").glob("renovation*")])
    runs = [(p.name, pd.read_csv(p / "per_floor.csv")) for p in cands if (p / "per_floor.csv").exists()]
    runs = [(n, d[d["status"] == "ok"]) for n, d in runs if "status" in d]
    runs = [(n, d) for n, d in runs if len(d) >= 5]
    if not runs:
        return []
    s = load_style()
    keys = [k for k in ["pred_score", "avg_shortest_path", "diameter", "closeness_mean", "max_betweenness", "n_dead_ends"] if all(f"after_{k}" in d for _, d in runs)]
    f, axes = plt.subplots(len(runs), len(keys), figsize=(s["figure"]["width_double"] * 1.1, 2.3 * len(runs) + 0.9), squeeze=False)
    for r, (name, d) in enumerate(runs):
        for c, k in enumerate(keys):
            ax = axes[r, c]
            b, a = d[f"before_{k}"].to_numpy(float), d[f"after_{k}"].to_numpy(float)
            ok = np.isfinite(b) & np.isfinite(a)
            b, a = b[ok], a[ok]
            better = BETTER.get(k, 0)
            imp = ((a - b) * better > 0) if better else np.zeros_like(a, bool)
            for bi, ai, im in zip(b, a, imp):
                ax.plot([0, 1], [bi, ai], color=s["palette"]["ours"] if im else "#BBBBBB", lw=0.7, alpha=0.65 if im else 0.5, zorder=1)
            ax.scatter([0] * len(b), b, s=9, color="#888", zorder=2); ax.scatter([1] * len(a), a, s=9, color="#888", zorder=2)
            ax.plot([0, 1], [b.mean(), a.mean()], "o-", color=s["palette"]["highlight"], ms=6, lw=2.0, zorder=3)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["现状", "更新"], fontsize=s["fonts"]["size_annot"]); ax.set_xlim(-0.35, 1.35)
            ax.tick_params(axis="y", labelsize=s["fonts"]["size_annot"] - 0.5)
            ttl = TOPO_LABEL.get(k, k) + (" ↑" if better > 0 else (" ↓" if better < 0 else ""))
            sub = f"均值 {b.mean():.2f} → {a.mean():.2f}" + (f"\n{imp.mean() * 100:.0f}% 楼层改善" if better else "\n（中性，仅报告）")
            ax.set_title(f"{ttl}\n{sub}", fontsize=s["fonts"]["size_annot"], linespacing=1.2, pad=4)
            ax.grid(axis="y", alpha=s["figure"]["grid_alpha"]); ax.spines[["top", "right"]].set_visible(False)
        axes[r, 0].text(-0.45, 0.5, f"{name}\nn = {len(d)}", transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=s["fonts"]["size_annot"] + 0.5, fontweight="bold")
    f.suptitle(title_for("R15") + "  — 每条灰 / 蓝线 = 一个真实楼层（蓝 = 改善），红点线 = 均值", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.03, 0, 1, 0.93))
    return savefig(f, out, "R15_stage3_renovation")


ALL = [*ALL, r15_renovation_batch]
