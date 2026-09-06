"""Process / method figures (F01–F07): framework, data overview, split protocol, retrieval funnel, generation examples."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from mall_space_planner.reporting.style import fig, label, load_style, savefig, title_for

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def _box(ax, x, y, w, h, text, fc, ec="#333", fs=9.5, bold=False, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color="#333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.1, color=color))


def _elbow(ax, pts, color="#333"):
    """Poly-line arrow through ``pts`` (list of (x, y)); arrow head on the last segment. Used to route around boxes."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.1, solid_capstyle="round")
    _arrow(ax, *pts[-2], *pts[-1], color=color)


def f01_framework(results: Path, out: Path) -> list[Path]:
    s = load_style()
    f, ax = plt.subplots(figsize=(s["figure"]["width_double"], 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.1)
    ax.axis("off")
    c1, c2, c3, c4 = "#E8F1FA", "#FDF2E3", "#E9F6F0", "#F3F3F3"
    _box(ax, 0.2, 4.3, 2.1, 1.0, "外部策划条件\n城市能级 · 人口经济\n商业竞争 · 项目体量", c4, fs=9)
    # stage 1
    ax.add_patch(FancyBboxPatch((2.7, 2.35), 4.1, 3.1, boxstyle="round,pad=0.02", fc="none", ec="#0072B2", lw=1.2, ls="--"))
    ax.text(2.8, 5.5, "阶段一  布局类型决策 + 可比案例检索", fontsize=10, color="#0072B2", fontweight="bold", va="bottom")
    _box(ax, 2.9, 4.3, 1.8, 0.9, "① 类型决策模型\n每种布局类型的\n预期评分 ± 区间", c1, fs=8.5)
    _box(ax, 4.95, 4.3, 1.7, 0.9, "② 设计师选型\n（默认取预期最高）", c1, fs=8.5)
    _box(ax, 2.9, 3.05, 1.8, 0.9, "③ 硬约束筛选\n同类城市 · 同面积档\n所选类型", c1, fs=8.5)
    _box(ax, 4.95, 3.05, 1.7, 0.9, "④ 相似案例检索\n+ 质量排序 + 解释", c1, fs=8.5)
    _box(ax, 3.4, 2.45, 2.8, 0.45, "输出：Top-K 拓扑原型 + 证据 + 反事实", "#FFFFFF", ec="#0072B2", fs=8.5)
    # stage 2
    ax.add_patch(FancyBboxPatch((7.1, 0.2), 2.75, 5.25, boxstyle="round,pad=0.02", fc="none", ec="#D55E00", lw=1.2, ls="--"))
    ax.text(7.2, 5.5, "阶段二  可控拓扑扩展 + 平面草案", fontsize=10, color="#D55E00", fontweight="bold", va="bottom")
    _box(ax, 7.3, 4.25, 2.35, 0.85, "⑤ 拓扑扩展\n规则 / 搜索 / 自回归图网络\n（骨架结构始终保持）", c2, fs=8.5)
    _box(ax, 7.3, 3.1, 2.35, 0.85, "⑥ 几何解码\n走廊缓冲 · 中庭 · 临街切分\n主力店 · 场地边界", c2, fs=8.5)
    _box(ax, 7.3, 1.95, 2.35, 0.85, "⑦ 修复与双重评估\n拓扑 5 项指标 + 几何检查", c2, fs=8.5)
    _box(ax, 7.3, 0.8, 2.35, 0.85, "输出：JSON / GeoJSON\nSVG / PNG 布局草案", "#FFFFFF", ec="#D55E00", fs=8.5)
    # bottom: data & evaluation
    _box(ax, 0.2, 0.3, 6.6, 1.5, "案例库：1 209 座商场 · 5 380 层平面拓扑\n城市 / 商圈 / 体量条件 · 公众综合评分\n"
         "评估：按商场分组的无泄漏划分 · 上 / 下界参照\n多随机种子 · 消融实验 · 与真实建成拓扑对比", c3, fs=8.6)
    # arrows
    _arrow(ax, 2.3, 4.8, 2.9, 4.75)
    _arrow(ax, 4.7, 4.75, 4.95, 4.75)
    _arrow(ax, 5.8, 4.3, 3.8, 3.95)
    _arrow(ax, 4.7, 3.5, 4.95, 3.5)
    _arrow(ax, 5.8, 3.05, 4.8, 2.9)
    _elbow(ax, [(6.2, 2.67), (6.95, 2.67), (6.95, 4.67), (7.3, 4.67)], color="#D55E00")  # routed through the gap between the two stages
    _arrow(ax, 8.47, 4.25, 8.47, 3.95, color="#D55E00")
    _arrow(ax, 8.47, 3.1, 8.47, 2.8, color="#D55E00")
    _arrow(ax, 8.47, 1.95, 8.47, 1.65, color="#D55E00")
    _arrow(ax, 3.5, 1.8, 3.5, 2.35, color="#009E73")
    _arrow(ax, 6.8, 1.05, 7.3, 1.2, color="#009E73")
    ax.text(0.2, 5.85, title_for("F01"), fontsize=12, fontweight="bold", va="bottom")
    return savefig(f, out, "F01_framework_overview")


def _cases(results: Path) -> pd.DataFrame | None:
    for p in [results / "data/cases.csv", Path("data/processed/legacy/cases.csv")]:
        if p.exists():
            return pd.read_csv(p)
    return None


def f02_data_overview(results: Path, out: Path) -> list[Path]:
    df = _cases(results)
    s = load_style()
    f, axes = plt.subplots(2, 2, figsize=(s["figure"]["width_double"], 5.2))
    (a1, a2), (a3, a4) = axes
    if df is None:
        for ax in axes.ravel():
            ax.axis("off")
        a1.text(0, 0.5, "未找到处理后的案例表（data/processed/legacy/cases.csv）。\n在 Mac 上运行 make_thesis_report.py 时会自动读取真实数据生成本图。", fontsize=9)
        f.suptitle(title_for("F02"), x=0.02, ha="left", fontweight="bold")
        return savefig(f, out, "F02_data_overview")
    # (a) score distribution per mall
    mall = df.groupby("mall_id").agg(score=("total_score", "first"), cluster=("city_cluster", "first"), area=("total_area", "first"), n_floors=("floor_id", "size")).dropna(subset=["score"])
    a1.hist(mall["score"], bins=np.arange(2.8, 5.01, 0.1), color=s["palette"]["ours"], edgecolor="white")
    a1.set_xlabel("商场公众综合评分")
    a1.set_ylabel("商场数")
    a1.set_title(f"(a) 评分分布（{len(mall)} 座商场，中位数 {mall['score'].median():.2f}）", loc="left", fontsize=s["fonts"]["size_label"])
    # (b) layout type × cluster
    lt = df.dropna(subset=["layout_type"])
    lt = lt[lt["layout_type"].isin(s["labels"]["layout_types"])]
    ct = pd.crosstab(lt["layout_type"], lt["city_cluster"]).reindex(s["labels"]["layout_types"]).fillna(0)
    bottom = np.zeros(len(ct))
    for i, c in enumerate(ct.columns):
        a2.bar(range(len(ct)), ct[c], bottom=bottom, color=s["palette"]["main"][i], label=label("clusters", str(int(c))).split("（")[0], edgecolor="white")
        bottom += ct[c].values
    a2.set_xticks(range(len(ct)))
    a2.set_xticklabels(ct.index, rotation=15)
    a2.set_ylabel("楼层数")
    a2.legend(fontsize=s["fonts"]["size_annot"])
    a2.set_title("(b) 布局类型 × 城市类别", loc="left", fontsize=s["fonts"]["size_label"])
    # (c) area
    a3.hist(np.log10(mall["area"].clip(lower=1000)), bins=30, color=s["palette"]["main"][2], edgecolor="white")
    for thr in (200_000, 450_000):
        a3.axvline(np.log10(thr), color="#444", ls="--", lw=1)
    a3.set_xticks([4, 4.5, 5, 5.5, 6])
    a3.set_xticklabels(["1 万", "3 万", "10 万", "30 万", "100 万"])
    a3.set_xlabel("项目总建筑面积 (m$^2$)")
    a3.set_ylabel("商场数")
    a3.set_title("(c) 体量分布（虚线 = 检索用面积档 20 / 45 万 m$^2$）", loc="left", fontsize=s["fonts"]["size_label"])
    # (d) graph size
    g = df.dropna(subset=["g_num_nodes"])
    a4.scatter(g["g_num_nodes"], g["g_num_edges"], s=5, alpha=0.35, color=s["palette"]["main"][3], lw=0)
    a4.set_xlabel("单层拓扑节点数（空间单元）")
    a4.set_ylabel("连接数")
    a4.set_title(f"(d) 楼层拓扑规模（{len(g)} 层，中位 {int(g['g_num_nodes'].median())} 节点）", loc="left", fontsize=s["fonts"]["size_label"])
    f.suptitle(title_for("F02"), x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "F02_data_overview")


def f03_condition_correlation(results: Path, out: Path) -> list[Path]:
    df = _cases(results)
    s = load_style()
    conds = ["total_area", "people", "GDP_2023", "PCDI_2023", "TP_2023", "mall_area_count", "nearest_distance_km", "count_1km", "count_2km", "Tx"]
    topo = ["g_num_nodes", "g_density", "g_avg_shortest_path", "g_num_cycles", "L2_integration", "L1_density", "L2_complexity"]
    if df is None:
        # fallback numbers from the audit (docs/experiments.md §0)
        vals = pd.Series({"total_area": 0.48, "PCDI_2023": 0.17, "GDP_2023": 0.15, "people": 0.13, "TP_2023": 0.14, "count_2km": 0.10, "count_1km": 0.08, "mall_area_count": 0.11, "Tx": 0.09, "nearest_distance_km": -0.06})
        topo_vals = pd.Series({"g_num_nodes": 0.30, "g_num_cycles": 0.28, "L2_integration": 0.22, "g_density": -0.25, "g_avg_shortest_path": 0.20, "L1_density": 0.18, "L2_complexity": 0.24})
        note = "（数值来自数据审计报告；本机运行时自动由案例表重算）"
    else:
        mall = df.groupby("mall_id").first()
        vals = pd.Series({c: mall[[c, "total_score"]].dropna().corr(method="spearman").iloc[0, 1] for c in conds if c in mall})
        fl = df.dropna(subset=["total_score"])
        topo_vals = pd.Series({c: fl[[c, "total_score"]].dropna().corr(method="spearman").iloc[0, 1] for c in topo if c in fl})
        note = ""
    f, (a1, a2) = plt.subplots(1, 2, figsize=(s["figure"]["width_double"], 3.0), sharex=True)
    for ax, ser, ttl, col in ((a1, vals, "(a) 外部策划条件（商场级）", s["palette"]["ours"]), (a2, topo_vals, "(b) 楼层拓扑指标", s["palette"]["main"][1])):
        ser = ser.sort_values()
        ax.barh(range(len(ser)), ser.values, color=[col if v >= 0 else s["palette"]["baseline"] for v in ser.values], edgecolor="white")
        ax.set_yticks(range(len(ser)))
        ax.set_yticklabels([label("conditions", k) if k in s["labels"]["conditions"] else {"g_num_nodes": "节点数", "g_density": "路网密度", "g_avg_shortest_path": "平均步行路径", "g_num_cycles": "环路数", "L2_integration": "整合度", "L1_density": "L1 密度", "L2_complexity": "拓扑复杂度"}.get(k, k) for k in ser.index])
        ax.axvline(0, color="#444", lw=0.9)
        ax.set_title(ttl, loc="left", fontsize=s["fonts"]["size_label"])
        ax.set_xlabel("与综合评分的秩相关系数")
        ax.grid(axis="y", alpha=0)
        lo, hi = min(ser.min(), 0), max(ser.max(), 0)
        ax.set_xlim(lo - 0.12, hi + 0.12)
        for i, v in enumerate(ser.values):
            # positive: label to the right of the bar; negative: label to the LEFT of the bar end (never over tick labels)
            ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=s["fonts"]["size_annot"])
        ax.tick_params(axis="y", pad=2)
    f.suptitle(title_for("F03") + note, x=0.02, ha="left", fontweight="bold")
    f.text(0.02, -0.02, "读法：+ 正相关 / − 负相关。体量与评分的关联最强；城市经济指标关联很弱甚至为负 → 评分主要由体量与商圈条件解释，布局类型是叠加其上的二阶因素。", fontsize=s["fonts"]["size_annot"], color="#555")
    f.tight_layout()
    return savefig(f, out, "F03_condition_score_correlation")


def f04_split_protocol(results: Path, out: Path) -> list[Path]:
    man = json.load(open(results / "data/manifest.json", encoding="utf-8")) if (results / "data/manifest.json").exists() else {}
    sp = man.get("split", {})
    s = load_style()
    f, ax = plt.subplots(figsize=(s["figure"]["width_double"], 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    tr, va, te = sp.get("train_groups", 967), sp.get("val_groups", 121), sp.get("test_groups", 121)
    tot = tr + va + te
    x = 0.3
    widths = [9.4 * n / tot for n in (tr, va, te)]
    widths = [max(w, 1.5) for w in widths]; widths[0] = 9.4 - widths[1] - widths[2]
    for w, name, c in zip(widths, (f"训练集：{tr} 座商场 / {sp.get('train_rows', 4318)} 层", f"验证集\n{va} 座 / {sp.get('val_rows', 551)} 层", f"测试集\n{te} 座 / {sp.get('test_rows', 526)} 层"), ("#CFE3F5", "#FBE3C4", "#CDEBDD")):
        _box(ax, x, 2.1, w, 0.9, name, c, fs=8.6)
        x += w
    ax.text(0.3, 1.75, "① 同一座商场的所有楼层只出现在一个子集中（按商场分组划分，按城市类别分层）→ 三个子集之间商场重叠 = 0", fontsize=9.2)
    ax.text(0.3, 1.35, "② 训练/评估时，每个楼层作为“待策划项目”，候选案例只取同城市类别、同面积档、且属于**其他商场**的楼层", fontsize=9.2)
    ax.text(0.3, 0.95, "③ 综合评分只作为学习目标与评估标准，绝不作为输入特征；上界 = 直接按评分排序，下界 = 随机排序", fontsize=9.2)
    ax.text(0.3, 0.55, "④ 阶段二：骨架→完整拓扑语料 5 632 条，最后 600 条为留出集，所有生成方法在同一留出集上比较", fontsize=9.2)
    ax.text(0.3, 0.1, "为什么：同一商场各层的评分与外部条件完全相同，若混入训练会让模型学会“抄同商场”而非学习规律（旧方案的主要问题之一）。", fontsize=8.8, color="#7F1D1D")
    ax.text(0.3, 3.1, title_for("F04"), fontsize=12, fontweight="bold", va="bottom")
    return savefig(f, out, "F04_split_protocol")


def f05_retrieval_funnel(results: Path, out: Path) -> list[Path]:
    s = load_style()
    f, ax = plt.subplots(figsize=(s["figure"]["width_double"], 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [("案例库全部楼层", 5114, "#E3E3E3"), ("同城市类别 + 同面积档", 900, "#CFE3F5"), ("所选布局类型", 260, "#B5D4EE"), ("相似度召回 Top-50", 50, "#8EBDE3"), ("质量排序 Top-10", 10, "#5A9FD4"), ("设计师选定原型", 1, "#0072B2")]
    maxw = 9.0
    y = 3.5
    for i, (name, n, c) in enumerate(stages):
        w = max(2.6, maxw * (np.log10(n + 1) / np.log10(5115)))
        _box(ax, (10 - w) / 2, y, w, 0.5, f"{name}   ≈ {n:,} 个" if n > 1 else name, c, fs=9, tc="white" if i >= 4 else "#111")
        y -= 0.62
    ax.text(0.2, 0.05, "说明：数量为示意量级；每一步都可追溯——系统记录被应用的约束、被放宽的约束、以及每个候选与输入条件最接近/差异最大的因素。", fontsize=8.6, color="#555")
    ax.text(0.2, 4.1, title_for("F05"), fontsize=12, fontweight="bold", va="bottom")
    return savefig(f, out, "F05_retrieval_funnel")


def _load_corpus(results: Path, n: int | None = 600):
    """Evaluation samples for the example figures. Preference: corpus v2 (real CSV-derived JSONL, test
    split) > v1 ShareGPT (held-out tail) > synthetic stand-in."""
    from mall_space_planner.data.corpus_builder import load_any_corpus
    from mall_space_planner.utils import resolve_config

    cfg = resolve_config("configs/data/legacy.yaml")
    v2 = [results / "data/stage2_corpus_v2.jsonl", Path(cfg.get("processed_dir", "data/processed/legacy")) / "stage2_corpus_v2.jsonl", Path(cfg.get("stage2_corpus") or "/nonexistent")]
    for p in v2:
        if p.exists():
            smp = load_any_corpus(p, split="test")
            return (smp[:n] if n else smp), p
    for p in [results / "data/sharegpt_data.json", Path(cfg.get("sharegpt_json") or "/nonexistent"), Path("data/samples/synthetic/sharegpt_sample.json")]:
        if p.exists():
            allsmp = load_any_corpus(p)
            return (allsmp[-n:] if n else allsmp), p
    return None, None


def _generators(results: Path):
    """rule, search16, ar_gnn best-of-16 (if a checkpoint is available) — built once."""
    from mall_space_planner.registry import build
    import mall_space_planner.stage2.generators  # noqa: F401

    rule = build("generator", {"name": "rule_expander", "params": {"label_style": "letters"}})
    search = build("generator", {"name": "search_expander", "params": {"n_trials": 16, "w_aspl": 1.5}})
    ar = None
    for ck in [results / "stage2/checkpoints/ar_gnn", Path("outputs/checkpoints/stage2/stage2_ar_gnn"), Path("/tmp/ck/stage2_ar_gnn")]:
        if (ck / "ar_gnn.pt").exists():
            try:
                ar = build("generator", {"name": "ar_gnn", "params": {"checkpoint": str(ck), "best_of": 16, "temperature": 0.7, "device": "cpu"}})
                break
            except Exception:  # noqa: BLE001
                ar = None
    return rule, search, ar


def _pick_examples(samples, per_type: int = 1, seed: int = 7, min_sk: int = 6, max_sk: int = 30):
    """One (or more) representative sample per layout type, medium size, connected target."""
    import networkx as nx
    from mall_space_planner.topology.convert import to_networkx

    rng = np.random.RandomState(seed)
    by = {}
    for s in samples:
        lt = s.layout_type.value if s.layout_type else None
        if not lt or lt == "Unknown_Layout":
            continue
        if not (min_sk <= s.skeleton.num_nodes <= max_sk) or s.target.num_nodes - s.skeleton.num_nodes < 6:
            continue
        if nx.number_connected_components(to_networkx(s.target)) != 1:
            continue
        by.setdefault(lt, []).append(s)
    order = load_style()["labels"]["layout_types"]
    picks = []
    for lt in order:
        pool = by.get(lt, [])
        if pool:
            idx = rng.choice(len(pool), size=min(per_type, len(pool)), replace=False)
            picks += [pool[i] for i in idx]
    return picks


def _gen_request(smp, seed=0):
    from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype
    from mall_space_planner.stage2.base import GenerationRequest

    n_t = smp.target_num_nodes or smp.target.num_nodes
    return GenerationRequest(prototype=TopologyPrototype(prototype_id=smp.sample_id, graph=smp.skeleton, layout_type=smp.layout_type), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=n_t, layout_type=smp.layout_type), seed=seed), n_t


def f06_generation_examples(results: Path, out: Path, per_type: int = 1) -> list[Path]:
    """One example per layout type (6 rows) x [skeleton, rule, rule+16, ours(+16), ground truth]."""
    from mall_space_planner.reporting.graphdraw import draw_topology, frame_of, full_layout, legend_handles, skeleton_layout
    from mall_space_planner.topology.convert import to_networkx
    from mall_space_planner.topology.metrics import attachment_overlap

    samples, _ = _load_corpus(results)
    if not samples:
        return []
    s = load_style()
    rule, search, ar = _generators(results)
    picks = _pick_examples(samples, per_type=per_type)
    cols = [("骨架\n（阶段一原型）", None), ("规则扩展", rule), ("规则\n+ 16 次择优", search), ("自回归图网络（本文）\n+ 16 次择优" if ar else "（无 AR-GNN 检查点）", ar), ("真实建成拓扑", "gt")]
    f, axes = plt.subplots(len(picks), len(cols), figsize=(s["figure"]["width_double"] * 1.15, 1.8 * len(picks) + 1.3))
    axes = np.atleast_2d(axes)
    for r, smp in enumerate(picks):
        sk_nodes = set(smp.skeleton.nodes)
        req, n_t = _gen_request(smp)
        sk_pos = skeleton_layout(to_networkx(smp.skeleton), seed=0)
        # lay out every panel of the row first so they can share one frame (same scale, same skeleton position)
        graphs, poses = [], []
        for ttl, gen in cols:
            g = smp.skeleton if gen is None else (smp.target if gen == "gt" else gen.generate(req, 0))
            G = to_networkx(g)
            graphs.append((g, G))
            poses.append(sk_pos if gen is None else full_layout(G, sk_nodes, sk_pos, seed=r))
        frame = frame_of(*poses)
        for c, ((ttl, gen), (g, G), pos) in enumerate(zip(cols, graphs, poses)):
            extra = ""
            if gen not in (None, "gt"):
                _, ap_ = attachment_overlap(smp.skeleton, smp.target, g)
                extra = f"\n分支位置正确率 {ap_:.0f}%"
            sub = f"{G.number_of_nodes()} 单元 / {G.number_of_edges()} 连接{extra}"
            draw_topology(axes[r, c], G, sk_nodes, pos=pos, title=sub, seed=r, size_ref_n=max(n_t, smp.target.num_nodes), frame=frame)
        axes[r, 0].text(-0.06, 0.5, f"{smp.layout_type.value}\n骨架 {smp.skeleton.num_nodes} → 目标 {n_t}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    f.legend(handles=legend_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    f.suptitle(title_for("F06") + "  — 每行一种布局类型；同一行各图中骨架位置与比例尺固定，便于对比", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0, 0.03, 1, 0.92))
    # column headers in a dedicated band between the suptitle and the first row (never collide with panel titles)
    for c, (ttl, _) in enumerate(cols):
        bb = axes[0, c].get_position()
        f.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.04, ttl, ha="center", va="bottom", linespacing=1.15, fontsize=s["fonts"]["size_label"], fontweight="bold", color=s["palette"]["ours"] if "本文" in ttl else "#222")
    return savefig(f, out, "F06_generation_examples")


def f06b_generation_gallery(results: Path, out: Path, per_type: int = 3, per_page: int = 6) -> list[Path]:
    """Gallery: more examples (``per_type`` per layout type), each = [skeleton | ours | truth] with a shared
    frame; two examples per row separated by a spacer column; ``per_page`` examples per page."""
    from mall_space_planner.reporting.graphdraw import draw_topology, frame_of, full_layout, legend_handles, skeleton_layout
    from mall_space_planner.topology.convert import to_networkx
    from mall_space_planner.topology.metrics import attachment_overlap

    samples, _ = _load_corpus(results)
    if not samples:
        return []
    s = load_style()
    rule, search, ar = _generators(results)
    gen = ar or search
    gen_name = "本文生成" if ar else "规则 + 择优生成"
    picks = _pick_examples(samples, per_type=per_type, seed=11, min_sk=5, max_sk=40)
    # order pages so that every page mixes layout types (type-major -> round-robin)
    by_type: dict[str, list] = {}
    for smp in picks:
        by_type.setdefault(smp.layout_type.value, []).append(smp)
    picks = [lst[i] for i in range(per_type) for lst in by_type.values() if i < len(lst)]
    written = []
    for page in range(0, len(picks), per_page):
        chunk = picks[page : page + per_page]
        nrow = (len(chunk) + 1) // 2
        f = plt.figure(figsize=(s["figure"]["width_double"] * 1.15, 2.0 * nrow + 0.9))
        gs = f.add_gridspec(nrow, 7, width_ratios=[1, 1, 1, 0.18, 1, 1, 1], hspace=0.55, wspace=0.12, left=0.02, right=0.98, top=0.90, bottom=0.07)
        for i, smp in enumerate(chunk):
            rr, cc = divmod(i, 2)
            base = 0 if cc == 0 else 4
            sk_nodes = set(smp.skeleton.nodes)
            req, n_t = _gen_request(smp)
            sk_pos = skeleton_layout(to_networkx(smp.skeleton), seed=0)
            g_ours = gen.generate(req, 0)
            ap = attachment_overlap(smp.skeleton, smp.target, g_ours)[1]
            G_ours, G_gt = to_networkx(g_ours), to_networkx(smp.target)
            p_ours = full_layout(G_ours, sk_nodes, sk_pos, seed=i)
            p_gt = full_layout(G_gt, sk_nodes, sk_pos, seed=i)
            frame = frame_of(sk_pos, p_ours, p_gt)
            panels = [
                (to_networkx(smp.skeleton), sk_pos, f"{smp.layout_type.value} · 骨架\n{smp.skeleton.num_nodes} 单元"),
                (G_ours, p_ours, f"{gen_name} {g_ours.num_nodes} 单元\n分支位置正确率 {ap:.0f}%"),
                (G_gt, p_gt, f"真实建成\n{smp.target.num_nodes} 单元"),
            ]
            for k, (G, pos, ttl) in enumerate(panels):
                ax = f.add_subplot(gs[rr, base + k])
                draw_topology(ax, G, sk_nodes, pos=pos, title=ttl, seed=i, size_ref_n=max(n_t, smp.target.num_nodes), frame=frame)
        f.legend(handles=legend_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=s["fonts"]["size_annot"])
        f.suptitle(f"{title_for('F06')}（补充示例 {page // per_page + 1}：骨架 → {gen_name} → 真实建成）", x=0.02, ha="left", fontweight="bold")
        written += savefig(f, out, f"F06b_generation_gallery_{page // per_page + 1}")
    return written


def f07_autoregressive_steps(results: Path, out: Path, n_steps: int = 7) -> list[Path]:
    """Teacher-forcing steps on one real sample: the target grows node by node in corpus order."""
    from mall_space_planner.reporting.graphdraw import draw_topology, legend_handles, skeleton_layout
    from mall_space_planner.stage2.generators.ar_gnn import canonical_order
    from mall_space_planner.topology.convert import to_networkx

    samples, _ = _load_corpus(results)
    if not samples:
        return []
    s = load_style()
    from mall_space_planner.reporting.graphdraw import frame_of, full_layout

    # a connected, medium-size example whose target grows by a good margin
    cands = [x for x in samples if 8 <= x.skeleton.num_nodes <= 14 and n_steps + 4 <= x.target.num_nodes - x.skeleton.num_nodes <= 22
             and nx.is_connected(to_networkx(x.skeleton)) and nx.is_connected(to_networkx(x.target))]
    smp = cands[3] if len(cands) > 3 else (cands[0] if cands else samples[0])
    tg = to_networkx(smp.target)
    order = canonical_order(smp.skeleton, smp.target, "label")
    sk_nodes = set(smp.skeleton.nodes)
    sk_pos = skeleton_layout(to_networkx(smp.skeleton), seed=0)
    pos = full_layout(tg, sk_nodes, sk_pos, seed=0)
    steps = min(n_steps, len(order))
    shown = list(smp.skeleton.nodes) + list(order[:steps])
    frame_steps = frame_of({k: pos[k] for k in shown})  # tight frame around what the step panels actually show
    ncol = 4
    nrow = int(np.ceil((steps + 2) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(s["figure"]["width_double"], 2.0 * nrow + 0.6))
    axes = axes.ravel()
    present = list(smp.skeleton.nodes)
    for i in range(steps + 1):
        hl = None
        ttl = "骨架（起点）"
        if i > 0:
            v = order[i - 1]
            present.append(v)
            nbrs = [u for u in tg.neighbors(v) if u in present[:-1]]
            hl = [(u, v) for u in nbrs]
            ttl = f"第 {i} 步：新单元 {v} 接到 {'、'.join(nbrs)}" + ("\n（形成环路）" if len(hl) > 1 else "")
        G = tg.subgraph(present).copy()
        draw_topology(axes[i], G, sk_nodes, pos={k: pos[k] for k in G.nodes}, title=ttl, highlight_edges=hl, seed=0, size_ref_n=len(shown), frame=frame_steps, node_scale=1.15)
    # final: full target in its own (larger) frame
    draw_topology(axes[steps + 1], tg, sk_nodes, pos=pos, title=f"…… 直到目标规模\n真实建成 {tg.number_of_nodes()} 单元", seed=0, size_ref_n=tg.number_of_nodes(), frame=frame_of(pos), node_scale=1.0)
    for ax in axes[steps + 2 :]:
        ax.axis("off")
    f.legend(handles=legend_handles() + [plt.Line2D([], [], color=s["palette"]["highlight"], lw=2.6, label="本步新建的连接（模型每步预测：接到哪里、是否再连第二条形成环路）")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    f.suptitle(title_for("F07") + f"（{smp.layout_type.value if smp.layout_type else ''}，按语料自身的生长顺序）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0, 0.06, 1, 0.97))
    return savefig(f, out, "F07_autoregressive_steps")


# --------------------------------------------------------------------------- stage-1 topology visuals
def f08_layout_type_gallery(results: Path, out: Path, per_type: int = 3) -> list[Path]:
    """What the six layout types look like as topologies (3 real examples each)."""
    from mall_space_planner.reporting.graphdraw import draw_topology
    from mall_space_planner.topology.convert import to_networkx

    samples, _ = _load_corpus(results, n=None)
    if not samples:
        return []
    s = load_style()
    types = s["labels"]["layout_types"]
    rng = np.random.RandomState(3)
    import networkx as nx
    f, axes = plt.subplots(len(types), per_type, figsize=(s["figure"]["width_double"], 1.55 * len(types) + 0.6))
    seen_malls: set = set()
    for r, lt in enumerate(types):
        pool = [x for x in samples if x.layout_type and x.layout_type.value == lt and 12 <= x.target.num_nodes <= 45 and nx.number_connected_components(to_networkx(x.target)) == 1]
        if len(pool) < per_type:  # sparse types: relax the size filter rather than leaving panels empty
            pool = [x for x in samples if x.layout_type and x.layout_type.value == lt and 8 <= x.target.num_nodes <= 60 and nx.number_connected_components(to_networkx(x.target)) == 1]
        # de-duplicate floors of the same mall / identical graphs so the row shows variety
        uniq, keys = [], set()
        for x in pool:
            k = (x.target.num_nodes, x.target.num_edges, getattr(x, "city", None))
            if k not in keys:
                keys.add(k); uniq.append(x)
        idx = rng.choice(len(uniq), size=min(per_type, len(uniq)), replace=False) if uniq else []
        for c in range(per_type):
            ax = axes[r, c]
            if c < len(idx):
                g = to_networkx(uniq[idx[c]].target)
                draw_topology(ax, g, set(), title=f"{g.number_of_nodes()} 单元 / {g.number_of_edges()} 连接", seed=c, node_scale=0.8)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "（该类型可用样本不足）", ha="center", va="center", transform=ax.transAxes, fontsize=s["fonts"]["size_annot"], color="#999")
        axes[r, 0].text(-0.04, 0.5, lt, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_label"], fontweight="bold", rotation=90)
    f.suptitle(title_for("F08") + "  — 节点 = 店铺 / 走廊段 / 中庭等空间单元，连线 = 直接相邻可达", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.03, 0, 1, 0.97))
    return savefig(f, out, "F08_layout_type_gallery")


def _service_for_worked_example(results: Path):
    """Build a PlanningService on the real DB if available, else synthetic; returns (svc, db, is_real)."""
    from mall_space_planner.api.service import PlanningService
    from mall_space_planner.data.case_db import CaseDatabase
    from mall_space_planner.utils import resolve_config

    s1 = resolve_config("configs/stage1/extra_trees.yaml")
    s2 = resolve_config("configs/stage2/search_baseline.yaml")
    real = Path(s1["data"]["processed_dir"])
    is_real = (real / "manifest.json").exists()
    if not is_real:
        real = Path("data/processed/synthetic")
        if not (real / "manifest.json").exists():
            return None, None, False
    db = CaseDatabase.load(str(real))
    s1["stage1"]["counterfactuals"] = {"enabled": False}
    s1["stage1"]["ranker"]["params"] = {**s1["stage1"]["ranker"].get("params", {}), "n_estimators": 120}
    return PlanningService(db, s1, s2), db, is_real


def f09_worked_example(results: Path, out: Path) -> list[Path]:
    """End-to-end on one query: type ranking -> Top-5 prototypes (drawn) -> generated topology -> floor-plan draft."""
    from mall_space_planner.reporting.graphdraw import draw_topology
    from mall_space_planner.schemas import ConstraintSet, PlanningCondition, SiteBoundary
    from mall_space_planner.topology.convert import to_networkx
    from matplotlib.patches import Polygon as MplPolygon

    svc, db, is_real = _service_for_worked_example(results)
    if svc is None:
        return []
    s = load_style()
    # a representative query: median conditions of cluster 2 in the DB
    df = db.cases
    med = df[df["city_cluster"] == 2][db.query_cols].median() if (df["city_cluster"] == 2).any() else df[db.query_cols].median()
    q = PlanningCondition(city_cluster=2, **{c: float(med[c]) for c in db.query_cols})
    types = svc.recommend_types(q)
    top_type = types.recommendations[0].layout_type if types and types.recommendations else None
    recs = svc.recommend_within_type(q, top_type, top_k=5) if top_type else svc.recommend(q, top_k=5, with_counterfactuals=False)
    proto = recs[0]
    n_sk = db.get_graph(proto.prototype_id).num_nodes
    n_t = max(int(n_sk * 1.8), n_sk + 8)
    gens = svc.generate(proto.prototype_id, SiteBoundary.rectangle(180, 120), ConstraintSet(target_num_nodes=n_t, target_num_shops=n_t, shop_area_min=60, shop_area_max=300), n_candidates=1, seed=0)
    layout, ev = gens[0]

    f = plt.figure(figsize=(s["figure"]["width_double"] * 1.15, 8.0))
    # rows: (a)+(b) | caption band | (c) prototypes | caption band | (d)+(e)
    gs = f.add_gridspec(5, 10, height_ratios=[1.05, 0.16, 0.85, 0.16, 1.6], hspace=0.35, wspace=0.6, left=0.04, right=0.98, top=0.93, bottom=0.08)
    # (a) conditions
    ax = f.add_subplot(gs[0, 0:3])
    ax.axis("off")
    def fmt(c):
        v = med[c]
        return f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
    lines = [f"{label('conditions', c)}：{fmt(c)}" for c in db.query_cols]
    ax.text(0, 1.10, "(a) 输入：待策划项目的外部条件", va="top", fontsize=s["fonts"]["size_label"], fontweight="bold")
    ax.text(0, 0.96, f"（{label('clusters', '2')} 的中位条件）", va="top", fontsize=s["fonts"]["size_annot"], color="#555")
    ax.text(0, 0.84, "\n".join(lines), va="top", fontsize=s["fonts"]["size_annot"], linespacing=1.45)
    # (b) type ranking
    ax = f.add_subplot(gs[0, 4:10])
    if types:
        rows = types.recommendations
        y = np.arange(len(rows))
        cols = [s["palette"]["highlight"] if r.rank == 1 else s["palette"]["ours"] for r in rows]
        ax.hlines(y, [r.ci_low for r in rows], [r.ci_high for r in rows], color=cols, lw=2.4)
        ax.scatter([r.expected_score for r in rows], y, color=cols, s=30, zorder=3)
        xmax = max(rr.ci_high for rr in rows); xmin = min(rr.ci_low for rr in rows)
        for yy, r in zip(y, rows):
            ax.text(xmax + (xmax - xmin) * 0.04, yy, f"可比案例 {r.n_comparable_cases} 个", va="center", fontsize=s["fonts"]["size_annot"], color="#555")
        ax.set_xlim(xmin - (xmax - xmin) * 0.08, xmax + (xmax - xmin) * 0.45)
        ax.set_yticks(y)
        ax.set_yticklabels([r.layout_type for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("期望综合评分（线段 = 80% 置信区间）")
        ax.set_title(f"(b) 阶段一 ①：每种布局类型的预期评分 → 首选「{top_type}」", loc="left", fontsize=s["fonts"]["size_label"], fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0)
    # (c) top-5 prototypes
    for i, r in enumerate(recs[:5]):
        ax = f.add_subplot(gs[2, 2 * i : 2 * i + 2])
        g = db.get_graph(r.prototype_id)
        if g is None:
            ax.axis("off")
            continue
        sim = f"\n相似度 {r.similarity:.2f}" if r.similarity is not None else ""
        draw_topology(ax, to_networkx(g), set(), title=f"#{r.rank}  评分 {r.quality_score:.2f}{sim}", seed=i, node_scale=0.75)
    # row captions live in dedicated spacer rows (never collide with panel titles)
    for row, txt in ((1, f"(c) 阶段一 ②–④：「{top_type}」内检索到的 Top-5 可比案例原型（同类城市 · 同面积档 · 其他商场；按预测质量排序）"),
                     (3, f"(d) 阶段二 ⑤：以 #1 为骨架扩展到 {layout.topology.num_nodes} 单元（大纲 5 项判据：{'全部合格' if ev.overall_pass else '未全部合格'}）　　(e) 阶段二 ⑥–⑦：180 m × 120 m 场地内的平面布局草案")):
        cap = f.add_subplot(gs[row, :])
        cap.axis("off")
        cap.text(0, 0.0, txt, ha="left", va="bottom", transform=cap.transAxes, fontsize=s["fonts"]["size_label"], fontweight="bold")
    # (d) generated topology + (e) plan
    ax = f.add_subplot(gs[4, 0:4])
    sk = db.get_graph(proto.prototype_id)
    draw_topology(ax, to_networkx(layout.topology), set(sk.nodes), title="黑 = 原型骨架，白 = 新增单元", seed=0)
    ax = f.add_subplot(gs[4, 4:10])
    ax.add_patch(MplPolygon(layout.boundary.exterior, closed=True, facecolor="#FAFAFA", edgecolor="#333", lw=1.4))
    kinds = {"shop": "#CFE8FF", "anchor": "#DCD6F7", "corridor": "#F0C987", "atrium": "#B5E7A0", "junction": "#222", "entrance": "#D9480F"}
    for u in layout.units:
        if u.polygon:
            ax.add_patch(MplPolygon(u.polygon, closed=True, facecolor=kinds.get(u.kind, "#ddd"), edgecolor="#666", lw=0.4, alpha=0.95))
    pos = layout.skeleton_positions
    for a, b in layout.topology.edges():
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#D9480F", lw=1.0, alpha=0.8)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    n_shop = sum(1 for u in layout.units if u.kind == "shop")
    ax.set_title(f"{n_shop} 个店铺单元 · 走廊沿拓扑连接展开", loc="left", fontsize=s["fonts"]["size_annot"] + 0.5)
    from matplotlib.patches import Patch
    f.legend(handles=[Patch(facecolor=v, edgecolor="#666", label=k2) for k2, v in [("店铺", kinds["shop"]), ("主力店", kinds["anchor"]), ("走廊", kinds["corridor"]), ("中庭", kinds["atrium"])]] + [plt.Line2D([], [], color="#D9480F", lw=1.2, label="拓扑连接")], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.005), fontsize=s["fonts"]["size_annot"])
    f.suptitle(title_for("F09") + ("" if is_real else "（合成数据演示；本机运行时自动使用真实案例库）"), x=0.02, ha="left", fontweight="bold")
    return savefig(f, out, "F09_worked_example")


def f10_retrieval_evidence(results: Path, out: Path) -> list[Path]:
    """Query vs. Top-3 retrieved prototypes: topology + standardised condition profile side by side."""
    from mall_space_planner.reporting.graphdraw import draw_topology
    from mall_space_planner.schemas import PlanningCondition
    from mall_space_planner.topology.convert import to_networkx

    svc, db, is_real = _service_for_worked_example(results)
    if svc is None:
        return []
    s = load_style()
    df = db.cases
    test = df[df["split"] == "test"] if "split" in df else df
    test = test[test["has_graph"] == True] if "has_graph" in test else test  # noqa: E712
    row = test.iloc[len(test) // 3]
    q = PlanningCondition(city_cluster=int(row["city_cluster"]), **{c: float(row[c]) for c in db.query_cols})
    recs = svc.recommend(q, top_k=3, with_counterfactuals=False)
    # standardised condition values (log1p for heavy tails) for query and recs
    cols = db.query_cols
    X = np.log1p(df[cols].clip(lower=0).astype(float))
    mu, sd = X.mean(), X.std().replace(0, 1)
    def z(r):
        return ((np.log1p(pd.Series({c: float(r[c]) for c in cols}).clip(lower=0)) - mu) / sd).values
    zq = z(row)
    f = plt.figure(figsize=(s["figure"]["width_double"] * 1.1, 5.2))
    gs = f.add_gridspec(2, 4, height_ratios=[1, 1.2], wspace=0.3, hspace=0.55, left=0.09, right=0.98, top=0.86, bottom=0.14)
    # top row: query real topology (this floor's actual) + 3 retrieved; three short title lines each
    gq = db.get_graph(row[db.id_col])
    ax = f.add_subplot(gs[0, 0])
    if gq is not None:
        draw_topology(ax, to_networkx(gq), set(), title=f"待策划项目\n（真实建成，仅作对照）\n{row.get('layout_type', '')} · 评分 {row[db.label_col]:.2f}", seed=0, node_scale=0.7)
    for i, r in enumerate(recs):
        ax = f.add_subplot(gs[0, i + 1])
        g = db.get_graph(r.prototype_id)
        rr = df[df[db.id_col] == r.prototype_id].iloc[0]
        if g is not None:
            city = str(rr.get("cityname", "") or "").strip()
            line3 = (city + " · " if city else "") + f"{rr['total_area']/1e4:.1f} 万 m$^2$"
            draw_topology(ax, to_networkx(g), set(), title=f"推荐 #{r.rank}\n{r.layout_type.value if r.layout_type else ''} · 评分 {r.quality_score:.2f}\n{line3}", seed=i, node_scale=0.7)
    # bottom: condition profile
    ax = f.add_subplot(gs[1, :])
    x = np.arange(len(cols))
    w = 0.2
    ax.bar(x - 1.5 * w, zq, w, color="#2B2B2B", label="待策划项目")
    for i, r in enumerate(recs):
        rr = df[df[db.id_col] == r.prototype_id].iloc[0]
        ax.bar(x + (i - 0.5) * w, z(rr), w, color=s["palette"]["main"][i], label=f"推荐 #{r.rank}")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    def _wrap(t: str, w: int = 5) -> str:
        t = t.replace(" ", "")
        return t if len(t) <= w else "\n".join(t[i : i + w] for i in range(0, len(t), w))
    ax.set_xticklabels([_wrap(label("conditions", c)) for c in cols], fontsize=s["fonts"]["size_annot"], linespacing=1.1)
    ax.set_ylabel("标准化后的条件值\n（0 = 全库平均）")
    ymax = max(abs(zq).max(), max(abs(z(df[df[db.id_col] == r.prototype_id].iloc[0])).max() for r in recs))
    ax.set_ylim(-ymax * 1.15, ymax * 1.55)  # head-room so the legend never sits on a bar
    ax.legend(ncol=4, fontsize=s["fonts"]["size_annot"], loc="upper left", frameon=False)
    ax.set_title("各项外部条件的对比：推荐案例在多数条件上与待策划项目接近（即“可比”）；差异大的条件会在解释中标为风险", loc="left", fontsize=s["fonts"]["size_annot"] + 0.5, pad=6)
    f.suptitle(title_for("F10") + ("" if is_real else "（合成数据演示）"), x=0.02, ha="left", fontweight="bold")
    return savefig(f, out, "F10_retrieval_evidence")


ALL = [f01_framework, f02_data_overview, f03_condition_correlation, f04_split_protocol, f05_retrieval_funnel, f06_generation_examples, f06b_generation_gallery, f07_autoregressive_steps, f08_layout_type_gallery, f09_worked_example, f10_retrieval_evidence]
