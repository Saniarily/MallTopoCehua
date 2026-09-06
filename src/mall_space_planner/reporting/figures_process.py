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
    _box(ax, 0.2, 0.3, 6.6, 1.5, "案例库：1 209 座商场 · 5 380 层平面拓扑 · 城市/商圈/体量条件 · 公众综合评分\n"
         "评估：按商场分组的无泄漏划分 · 上/下界参照 · 多随机种子 · 消融实验 · 与真实建成拓扑对比", c3, fs=8.8)
    # arrows
    _arrow(ax, 2.3, 4.8, 2.9, 4.75)
    _arrow(ax, 4.7, 4.75, 4.95, 4.75)
    _arrow(ax, 5.8, 4.3, 3.8, 3.95)
    _arrow(ax, 4.7, 3.5, 4.95, 3.5)
    _arrow(ax, 5.8, 3.05, 4.8, 2.9)
    _arrow(ax, 6.2, 2.67, 7.3, 4.6, color="#D55E00")
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
    a3.set_xlabel("项目总建筑面积 (m²)")
    a3.set_ylabel("商场数")
    a3.set_title("(c) 体量分布（虚线 = 检索用面积档 20/45 万 m²）", loc="left", fontsize=s["fonts"]["size_label"])
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
        for i, v in enumerate(ser.values):
            ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=s["fonts"]["size_annot"])
    a1.set_xlim(-0.2, 0.6)
    f.suptitle(title_for("F03") + note, x=0.02, ha="left", fontweight="bold")
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


def _load_corpus(results: Path, n: int = 600):
    from mall_space_planner.data.sharegpt_adapter import load_sharegpt
    from mall_space_planner.utils import resolve_config

    for p in [results / "data/sharegpt_data.json", Path(resolve_config("configs/data/legacy.yaml").get("sharegpt_json") or "/nonexistent"), Path("data/samples/synthetic/sharegpt_sample.json")]:
        if p.exists():
            return load_sharegpt(p)[-n:], p
    return None, None


def _draw_graph(ax, g: nx.Graph, pos, skeleton_nodes: set, title: str, highlight_edges=None, node_size=42):
    s = load_style()
    sk_e = [e for e in g.edges if e[0] in skeleton_nodes and e[1] in skeleton_nodes]
    new_e = [e for e in g.edges if e not in sk_e and (e[1], e[0]) not in sk_e]
    nx.draw_networkx_edges(g, pos, edgelist=sk_e, ax=ax, width=1.8, edge_color="#333")
    nx.draw_networkx_edges(g, pos, edgelist=new_e, ax=ax, width=1.1, edge_color=s["palette"]["ours"], alpha=0.9)
    if highlight_edges:
        nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, ax=ax, width=2.2, edge_color=s["palette"]["highlight"])
    sk = [n for n in g.nodes if n in skeleton_nodes]
    nw = [n for n in g.nodes if n not in skeleton_nodes]
    nx.draw_networkx_nodes(g, pos, nodelist=sk, ax=ax, node_size=node_size, node_color="#333", linewidths=0)
    nx.draw_networkx_nodes(g, pos, nodelist=nw, ax=ax, node_size=node_size * 0.8, node_color=s["palette"]["ours"], linewidths=0.6, edgecolors="white")
    ax.set_title(title, fontsize=s["fonts"]["size_label"], loc="left")
    ax.axis("off")


def f06_generation_examples(results: Path, out: Path, n_examples: int = 3) -> list[Path]:
    from mall_space_planner.registry import build
    from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype
    from mall_space_planner.stage2.base import GenerationRequest
    from mall_space_planner.topology.convert import to_networkx
    import mall_space_planner.stage2.generators  # noqa: F401

    samples, _ = _load_corpus(results)
    if not samples:
        return []
    s = load_style()
    rule = build("generator", {"name": "rule_expander", "params": {"label_style": "letters"}})
    ar = None
    for ck in [results / "stage2/checkpoints/ar_gnn", Path("outputs/checkpoints/stage2/stage2_ar_gnn")]:
        if (ck / "ar_gnn.pt").exists():
            try:
                ar = build("generator", {"name": "ar_gnn", "params": {"checkpoint": str(ck), "best_of": 16, "temperature": 0.7, "device": "cpu"}})
                break
            except Exception:  # noqa: BLE001
                ar = None
    cols = ["骨架（阶段一原型）", "规则扩展", "自回归图网络（本文）" if ar else "规则扩展 + 16 次择优", "真实建成拓扑"]
    gen2 = ar or build("generator", {"name": "search_expander", "params": {"n_trials": 16}})
    rng = np.random.RandomState(7)
    picks = [samples[i] for i in rng.choice(len(samples), size=min(n_examples, len(samples)), replace=False)]
    f, axes = plt.subplots(len(picks), 4, figsize=(s["figure"]["width_double"], 1.9 * len(picks) + 0.4))
    axes = np.atleast_2d(axes)
    for r, smp in enumerate(picks):
        sk_nodes = set(smp.skeleton.nodes)
        n_t = smp.target_num_nodes or smp.target.num_nodes
        req = GenerationRequest(prototype=TopologyPrototype(prototype_id=smp.sample_id, graph=smp.skeleton, layout_type=smp.layout_type), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=n_t, layout_type=smp.layout_type), seed=0)
        graphs = [smp.skeleton, rule.generate(req, 0), gen2.generate(req, 0), smp.target]
        base_pos = nx.kamada_kawai_layout(to_networkx(smp.target))
        for c, (g, ttl) in enumerate(zip(graphs, cols)):
            G = to_networkx(g)
            pos = nx.spring_layout(G, pos={n: base_pos[n] for n in G.nodes if n in base_pos}, fixed=[n for n in G.nodes if n in sk_nodes] or None, seed=0, k=0.35)
            _draw_graph(axes[r, c], G, pos, sk_nodes, f"{ttl}\n{G.number_of_nodes()} 节点 / {G.number_of_edges()} 连接" if r == 0 else f"{G.number_of_nodes()} 节点 / {G.number_of_edges()} 连接")
        axes[r, 0].text(-0.05, 0.5, f"{smp.layout_type.value if smp.layout_type else ''}\n目标 {n_t} 节点", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"])
    f.suptitle(title_for("F06") + "  （黑 = 骨架节点与连接，蓝 = 新增）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout()
    return savefig(f, out, "F06_generation_examples")


def f07_autoregressive_steps(results: Path, out: Path, n_steps: int = 6) -> list[Path]:
    """Teacher-forcing steps on one real sample: shows how the target grows node by node in corpus order."""
    from mall_space_planner.stage2.generators.ar_gnn import canonical_order
    from mall_space_planner.topology.convert import to_networkx

    samples, _ = _load_corpus(results)
    if not samples:
        return []
    s = load_style()
    smp = next((x for x in samples if 8 <= x.skeleton.num_nodes <= 12 and x.target.num_nodes - x.skeleton.num_nodes >= n_steps), samples[0])
    tg = to_networkx(smp.target)
    order = canonical_order(smp.skeleton, smp.target, "label")
    sk_nodes = set(smp.skeleton.nodes)
    pos = nx.kamada_kawai_layout(tg)
    steps = min(n_steps, len(order))
    f, axes = plt.subplots(1, steps + 1, figsize=(s["figure"]["width_double"], 2.1))
    present = list(smp.skeleton.nodes)
    for i in range(steps + 1):
        G = tg.subgraph(present).copy()
        hl = None
        if i > 0:
            v = order[i - 1]
            hl = [(u, v) for u in G.neighbors(v)]
        _draw_graph(axes[i], G, {n: pos[n] for n in G.nodes}, sk_nodes, "骨架" if i == 0 else f"第 {i} 步：加入 {order[i-1]}", highlight_edges=hl, node_size=30)
        if i < steps:
            present.append(order[i])
    f.suptitle(title_for("F07") + "  （红 = 本步新建的连接；模型每一步预测“新单元接到哪里、是否形成环路”）", x=0.02, ha="left", fontweight="bold", fontsize=s["fonts"]["size_title"])
    f.tight_layout()
    return savefig(f, out, "F07_autoregressive_steps")


ALL = [f01_framework, f02_data_overview, f03_condition_correlation, f04_split_protocol, f05_retrieval_funnel, f06_generation_examples, f07_autoregressive_steps]
