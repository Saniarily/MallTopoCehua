"""M-series figures: real-data *case* figures (≥ 3 malls of different layout types per figure) that show, for every
method section of the thesis, what the architecture does and what changes between method variants / ablations / rule
versions – as topology floor plans an architect can read, not as metric bars.

Every figure is a standalone function ``mNN_*(results, out) -> list[Path]`` (same contract as ``figures_process``).
Output = png + pdf + svg (text stays editable, see ``reporting.style``). Inputs are resolved in this order:

* real data on the Mac (``configs/data/legacy.yaml``: processed case DB, corpus v2 JSONL, graph CSV dir, masks) and the
  AR-GNN checkpoints (``outputs/checkpoints/stage2/*`` or ``<results>/stage2/checkpoints/*``);
* otherwise the sandbox stand-ins (synthetic DB, fixture floor ``B000A0E928_1``) so the *code path* is verified before
  the user runs it; such figures are watermarked "沙箱示意（合成 / 样例数据）".

Case selection is deterministic (seeded) and prefers one floor per layout type, medium size, connected target, so the
same floors recur across figures (the reader follows the same three or four malls through all stages).

Figure list (code → section):

  M01  §5.1–5.3  阶段一流程：3 个不同城市类别的真实查询 → 类型期望评分 → 类型内 Top-3 原型（骨架图）
  M02  §5.4      阶段一排序器比较：同一查询下 4 种排序器的 Top-3 原型（规则 / 极端随机树 / LambdaMART / 深度残差）
  M03  §5.2      类型条件效应：同一组条件在三类城市下的类型排序变化
  M04  §6.2–6.3  阶段二方法比较：骨架 → 规则 / 规则+16 / AR-GNN / AR-GNN+16 / 真实，每行一种布局类型（固定骨架位置）
  M05  §6.4      阶段二消融：完整模型 vs 单接点标签 vs BFS 顺序 vs 贪心顺序（同一骨架，4 行类型）
  M06  §6.5      解码策略：贪心 τ=0 / 采样 τ=0.7 / 采样+16 择优（同一骨架）
  M07  §6.3      AR-GNN 逐步生长（真实样本，教师强制顺序）与多接点闭环
  M08  §7.3–7.5  阶段三拟合流程：轮廓 → 内缩区 + 中轴 → 初始化（外环钉在内缩边界）→ 松弛后 → 成廊，3 个真实楼层
  M09  §7.5      规则前后：开角 / 断头→出入口 / 边越界修复（关 vs 开），同一楼层
  M10  §7.6      迁移到新轮廓：同一网络放入自身轮廓 / 另一真实轮廓 / 手绘 L 形（边不越界）
  M11  §8.4.4    改造工作流：v1（锚定全部骨架）vs v2（仅锚定出入口 + 评分）vs v2b（回路中性），3 个真实楼层
  M12  §3.2      数据层级：色块平面 + 完整 M 网络 + 骨架（真实楼层，3 种类型）
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from mall_space_planner.reporting.style import label, load_style, savefig

warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")

SANDBOX_NOTE = "沙箱示意（合成 / 样例数据）— 本机运行时自动使用真实案例库"
TYPES_ORDER = ["一字型", "简单环型", "多环型", "简单型", "简单集中型", "复杂集中型"]


# ============================================================================================ shared context
@dataclass
class Ctx:
    """Everything the M figures need, resolved once per run."""

    results: Path
    db: Any = None  # CaseDatabase
    is_real_db: bool = False
    samples: list = field(default_factory=list)  # Stage-2 corpus (test split)
    corpus_path: Path | None = None
    is_real_corpus: bool = False
    ds: Any = None  # Stage3Dataset
    graph_dir: Path | None = None
    is_real_graphs: bool = False
    svc: Any = None  # PlanningService (lazy)
    gens: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ---- generators -----------------------------------------------------------------------------------------
    def generator(self, name: str):  # noqa: ANN201
        """rule | search16 | ar | ar16 | ar_greedy | ar_single | ar_bfs | ar_greedy_order (None when no checkpoint)."""
        if name in self.gens:
            return self.gens[name]
        from mall_space_planner.registry import build
        import mall_space_planner.stage2.generators  # noqa: F401

        g = None
        if name == "rule":
            g = build("generator", {"name": "rule_expander", "params": {"label_style": "letters"}})
        elif name == "search16":
            g = build("generator", {"name": "search_expander", "params": {"n_trials": 16, "w_aspl": 1.5}})
        else:
            ck_name = {"ar": "stage2_ar_gnn", "ar16": "stage2_ar_gnn", "ar_greedy": "stage2_ar_gnn", "ar_single": "stage2_ar_gnn_single_label",
                       "ar_bfs": "stage2_ar_gnn_bfs_order", "ar_greedy_order": "stage2_ar_gnn_greedy_order"}[name]
            extra = {"ar": {}, "ar16": {"best_of": 16}, "ar_greedy": {"temperature": 0.0}, "ar_single": {"loss_mode": "single"},
                     "ar_bfs": {"order": "bfs"}, "ar_greedy_order": {"order": "greedy"}}[name]
            for ck in [Path("outputs/checkpoints/stage2") / ck_name, self.results / "stage2/checkpoints" / ck_name.replace("stage2_", ""), Path("/tmp/ck") / ck_name]:
                if (ck / "ar_gnn.pt").exists():
                    try:
                        g = build("generator", {"name": "ar_gnn", "params": {"checkpoint": str(ck), "temperature": 0.7, "device": "cpu", **extra}})
                        break
                    except Exception as exc:  # noqa: BLE001
                        self.notes.append(f"checkpoint {ck} unusable: {exc}")
        self.gens[name] = g
        return g

    # ---- stage 1 service -------------------------------------------------------------------------------------
    def service(self):  # noqa: ANN201
        if self.svc is not None or self.db is None:
            return self.svc
        from mall_space_planner.api.service import PlanningService
        from mall_space_planner.utils import resolve_config

        s1 = resolve_config("configs/stage1/extra_trees.yaml")
        s2 = resolve_config("configs/stage2/search_baseline.yaml")
        s1["stage1"]["counterfactuals"] = {"enabled": False}
        s1["stage1"]["ranker"]["params"] = {**s1["stage1"]["ranker"].get("params", {}), "n_estimators": 150}
        self.svc = PlanningService(self.db, s1, s2)
        return self.svc

    def pipeline_for(self, ranker_cfg: str):  # noqa: ANN201
        """A Stage-1 pipeline fitted with another ranker config (for the ranker-comparison figure)."""
        from mall_space_planner.stage1.pipelines.recommend import Stage1Pipeline
        from mall_space_planner.utils import resolve_config

        cfg = resolve_config(ranker_cfg)
        cfg["stage1"]["counterfactuals"] = {"enabled": False}
        p = cfg["stage1"]["ranker"].get("params", {})
        if cfg["stage1"]["ranker"]["name"] == "deep_residual":
            p = {**p, "epochs": 8, "ensemble_k": 1}
        elif "n_estimators" in p or cfg["stage1"]["ranker"]["name"] in ("extra_trees", "random_forest"):
            p = {**p, "n_estimators": 150}
        cfg["stage1"]["ranker"]["params"] = p
        return Stage1Pipeline(cfg, self.db).fit()


def build_ctx(results: Path) -> Ctx:
    from mall_space_planner.data.case_db import CaseDatabase
    from mall_space_planner.data.corpus_builder import load_any_corpus
    from mall_space_planner.stage3.dataset import Stage3Dataset, Stage3Paths
    from mall_space_planner.utils import resolve_config

    ctx = Ctx(results=results)
    cfg = resolve_config("configs/data/legacy.yaml")
    # case DB
    real = Path(cfg.get("processed_dir", "data/processed/legacy"))
    if (real / "manifest.json").exists():
        ctx.db, ctx.is_real_db = CaseDatabase.load(str(real)), True
    elif (Path("data/processed/synthetic") / "manifest.json").exists():
        ctx.db, ctx.is_real_db = CaseDatabase.load("data/processed/synthetic"), False
    # stage-2 corpus (test split)
    for p in [results / "data/stage2_corpus_v2.jsonl", real / "stage2_corpus_v2.jsonl", Path(cfg.get("stage2_corpus") or "/nonexistent")]:
        if p.exists():
            ctx.samples, ctx.corpus_path, ctx.is_real_corpus = load_any_corpus(p, split="test"), p, True
            break
    if not ctx.samples:
        for p in [Path("data/samples/synthetic/sharegpt_sample.json")]:
            if p.exists():
                allsmp = load_any_corpus(p)
                ctx.samples, ctx.corpus_path, ctx.is_real_corpus = allsmp[-600:], p, False
    # stage-3 dataset + graph dir
    paths = Stage3Paths.from_config(cfg)
    for d in [paths.graph_dir, results / "data/graph_csv", Path("tests/fixtures/graph_csv")]:
        if d and d.exists() and any(d.glob("*_total.csv")):
            paths.graph_dir = d
            ctx.graph_dir = d
            ctx.is_real_graphs = d == Stage3Paths.from_config(cfg).graph_dir
            break
    ctx.ds = Stage3Dataset(paths)
    return ctx


# ============================================================================================ helpers
def _note(f, ctx: Ctx, real: bool) -> None:  # noqa: ANN001
    if not real:
        f.text(0.995, 0.005, SANDBOX_NOTE, ha="right", va="bottom", fontsize=7.5, color="#B00020", alpha=0.9)


def _pick_cases(samples, per_type: int = 1, seed: int = 7, min_sk: int = 6, max_sk: int = 30, min_growth: int = 6, types: list[str] | None = None):  # noqa: ANN001
    """Deterministic: ``per_type`` samples per layout type (medium skeleton, connected target)."""
    from mall_space_planner.topology.convert import to_networkx

    rng = np.random.RandomState(seed)
    by: dict[str, list] = {}
    for s in samples:
        lt = s.layout_type.value if s.layout_type else None
        if not lt or lt == "Unknown_Layout":
            continue
        if not (min_sk <= s.skeleton.num_nodes <= max_sk) or s.target.num_nodes - s.skeleton.num_nodes < min_growth:
            continue
        if nx.number_connected_components(to_networkx(s.target)) != 1:
            continue
        by.setdefault(lt, []).append(s)
    picks = []
    for lt in types or TYPES_ORDER:
        pool = by.get(lt, [])
        if pool:
            idx = rng.choice(len(pool), size=min(per_type, len(pool)), replace=False)
            picks += [pool[i] for i in idx]
    return picks


def _req(smp, seed: int = 0):  # noqa: ANN001
    from mall_space_planner.schemas import ConstraintSet, SiteBoundary, TopologyPrototype
    from mall_space_planner.stage2.base import GenerationRequest

    n_t = smp.target_num_nodes or smp.target.num_nodes
    return GenerationRequest(prototype=TopologyPrototype(prototype_id=smp.sample_id, graph=smp.skeleton, layout_type=smp.layout_type), boundary=SiteBoundary.rectangle(100, 100), constraints=ConstraintSet(target_num_nodes=n_t, layout_type=smp.layout_type), seed=seed), n_t


def _row_label(ax, smp, extra: str = "") -> None:  # noqa: ANN001
    s = load_style()
    lt = smp.layout_type.value if smp.layout_type else ""
    ax.text(-0.06, 0.5, f"{lt}\n{smp.sample_id}\n骨架 {smp.skeleton.num_nodes} → 目标 {smp.target.num_nodes}{extra}", transform=ax.transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90, linespacing=1.2)


def _col_headers(f, axes_row, cols: list[str], dy: float = 0.035) -> None:  # noqa: ANN001
    s = load_style()
    for c, ttl in enumerate(cols):
        bb = axes_row[c].get_position()
        f.text((bb.x0 + bb.x1) / 2, bb.y1 + dy, ttl, ha="center", va="bottom", linespacing=1.15, fontsize=s["fonts"]["size_label"], fontweight="bold", color=s["palette"]["ours"] if "本文" in ttl else "#222")


def _generation_grid(ctx: Ctx, out: Path, stem: str, code: str, cols: list[tuple[str, str | None]], picks, suptitle_extra: str, per_panel_metric: bool = True, seed_offset: int = 0):  # noqa: ANN001
    """Rows = cases, columns = [skeleton, generators..., ground truth]; shared frame per row."""
    from mall_space_planner.reporting.graphdraw import draw_topology, frame_of, full_layout, legend_handles, skeleton_layout
    from mall_space_planner.topology.convert import to_networkx
    from mall_space_planner.topology.metrics import attachment_overlap, degree_histogram_emd

    s = load_style()
    f, axes = plt.subplots(len(picks), len(cols), figsize=(s["figure"]["width_double"] * (0.95 + 0.22 * len(cols)), 1.85 * len(picks) + 1.3))
    axes = np.atleast_2d(axes)
    for r, smp in enumerate(picks):
        sk_nodes = set(smp.skeleton.nodes)
        req, n_t = _req(smp, seed_offset)
        sk_pos = skeleton_layout(to_networkx(smp.skeleton), seed=0)
        graphs, poses, subs = [], [], []
        for ttl, gname in cols:
            if gname is None:
                g = smp.skeleton
            elif gname == "gt":
                g = smp.target
            else:
                gen = ctx.generator(gname)
                g = gen.generate(req, seed_offset) if gen is not None else None
            if g is None:
                graphs.append(None); poses.append(None); subs.append("（无检查点）"); continue
            G = to_networkx(g)
            graphs.append((g, G))
            poses.append(sk_pos if gname is None else full_layout(G, sk_nodes, sk_pos, seed=r))
            extra = ""
            if gname not in (None, "gt") and per_panel_metric:
                rc, pr = attachment_overlap(smp.skeleton, smp.target, g)
                emd = degree_histogram_emd(smp.target, g)
                extra = f"\n分支位置正确率 {pr:.0f}% · 度分布差异 {emd:.2f}"
            subs.append(f"{G.number_of_nodes()} 单元 / {G.number_of_edges()} 连接{extra}")
        frame = frame_of(*[p for p in poses if p])
        for c, ((ttl, gname), gg, pos, sub) in enumerate(zip(cols, graphs, poses, subs)):
            ax = axes[r, c]
            if gg is None:
                ax.axis("off"); ax.text(0.5, 0.5, sub, ha="center", va="center", fontsize=8, color="#888", transform=ax.transAxes); continue
            draw_topology(ax, gg[1], sk_nodes, pos=pos, title=sub, seed=r, size_ref_n=max(n_t, smp.target.num_nodes), frame=frame)
        _row_label(axes[r, 0], smp)
    f.legend(handles=legend_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    f.suptitle(f"{code}  {suptitle_extra}", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0, 0.03, 1, 0.92))
    _col_headers(f, axes[0], [t for t, _ in cols])
    _note(f, ctx, ctx.is_real_corpus)
    return savefig(f, out, stem)


# ============================================================================================ Stage 1
def _conditions_text(db, row) -> str:  # noqa: ANN001
    parts = []
    for c in db.query_cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        parts.append(f"{label('conditions', c)}：{v:,.0f}" if abs(float(v)) >= 100 else f"{label('conditions', c)}：{float(v):.2f}")
    return "\n".join(parts)


def _query_from_row(db, row):  # noqa: ANN001
    from mall_space_planner.schemas import PlanningCondition

    return PlanningCondition(city_cluster=int(row["city_cluster"]) if pd.notna(row.get("city_cluster")) else None, **{c: (float(row[c]) if pd.notna(row.get(c)) else None) for c in db.query_cols})


def _pick_query_floors(db, n: int = 3, seed: int = 3) -> list[pd.Series]:  # noqa: ANN001
    """One test floor per city cluster (different layout types when possible), with a graph and a score."""
    df = db.split("test") if "split" in db.cases else db.cases
    df = df[df.get("has_graph", True) & df[db.label_col].notna()] if "has_graph" in df else df[df[db.label_col].notna()]
    rng = np.random.RandomState(seed)
    out, used_types = [], set()
    clusters = sorted(df["city_cluster"].dropna().unique()) if "city_cluster" in df else [None]
    for cl in clusters[:n]:
        sub = df[df["city_cluster"] == cl] if cl is not None else df
        if "layout_type" in sub:
            fresh = sub[~sub["layout_type"].astype(str).isin(used_types) & sub["layout_type"].notna()]
            sub = fresh if len(fresh) else sub
        if len(sub) == 0:
            continue
        r = sub.iloc[int(rng.randint(len(sub)))]
        out.append(r)
        if "layout_type" in r and pd.notna(r.get("layout_type")):
            used_types.add(str(r["layout_type"]))
    while len(out) < n and len(df) > len(out):
        out.append(df.iloc[int(rng.randint(len(df)))])
    return out[:n]


def _draw_type_ranking(ax, types, title: str) -> None:  # noqa: ANN001
    s = load_style()
    rows = types.recommendations
    y = np.arange(len(rows))
    cols = [s["palette"]["highlight"] if r.rank == 1 else s["palette"]["ours"] for r in rows]
    for yi, r, c in zip(y, rows, cols):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter([r.expected_score], [yi], s=34, color=c, zorder=5)
        ax.text(ax.get_xlim()[1] if False else r.ci_high, yi, f"  可比 {r.n_comparable_cases}", va="center", fontsize=s["fonts"]["size_annot"] - 1, color="#666")
    ax.set_yticks(y); ax.set_yticklabels([r.layout_type for r in rows], fontsize=s["fonts"]["size_annot"])
    ax.invert_yaxis()
    ax.set_xlabel("期望综合评分（线段 = 80% 置信区间）", fontsize=s["fonts"]["size_annot"])
    ax.tick_params(axis="x", labelsize=s["fonts"]["size_annot"])
    ax.grid(axis="x", alpha=s["figure"]["grid_alpha"]); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=s["fonts"]["size_annot"] + 0.5, loc="left")


def m01_stage1_pipeline_cases(ctx: Ctx, out: Path) -> list[Path]:
    """3 real queries (one per city cluster) → type ranking → Top-3 prototypes within the first-ranked type."""
    from mall_space_planner.reporting.graphdraw import draw_topology, legend_handles

    if ctx.db is None:
        return []
    svc = ctx.service()
    s = load_style()
    rows = _pick_query_floors(ctx.db, 3)
    f = plt.figure(figsize=(s["figure"]["width_double"] * 1.25, 2.75 * len(rows) + 0.9))
    gs = f.add_gridspec(len(rows), 7, width_ratios=[1.25, 1.7, 0.08, 1, 1, 1, 0.05], wspace=0.25, hspace=0.55, left=0.02, right=0.99, top=0.90, bottom=0.07)
    for i, r in enumerate(rows):
        q = _query_from_row(ctx.db, r)
        types = svc.recommend_types(q)
        top = types.recommendations[0].layout_type if types and types.recommendations else None
        mall = str(r[ctx.db.mall_id_col])
        recs = svc.stage1.recommend(q.model_copy(update={"preferred_layout": __import__("mall_space_planner.schemas", fromlist=["LayoutType"]).LayoutType(top)}) if top else q, top_k=3, exclude_mall_ids={mall}, with_explanations=False, with_counterfactuals=False)
        # (a) conditions
        ax = f.add_subplot(gs[i, 0]); ax.axis("off")
        cl = int(r["city_cluster"]) if pd.notna(r.get("city_cluster")) else None
        hdr = f"案例 {i + 1}：{label('clusters', str(cl)) if cl else ''}\n查询 = 商场 {mall} 的真实条件" + (f"\n实际建成类型：{r['layout_type']}，评分 {float(r[ctx.db.label_col]):.1f}" if pd.notna(r.get("layout_type")) else "")
        ax.text(0, 1.0, hdr, va="top", fontsize=s["fonts"]["size_annot"] + 0.5, fontweight="bold", linespacing=1.3)
        ax.text(0, 0.70, _conditions_text(ctx.db, r), va="top", fontsize=s["fonts"]["size_annot"] - 0.5, linespacing=1.4)
        # (b) type ranking
        ax = f.add_subplot(gs[i, 1])
        if types:
            _draw_type_ranking(ax, types, f"① 六种布局类型的期望评分（红 = 首选：{top}）")
        # (c) Top-3 prototypes (skeletons)
        for j in range(3):
            ax = f.add_subplot(gs[i, 3 + j])
            if j < len(recs):
                rec = recs[j]
                g = ctx.db.get_graph(rec.prototype_id)
                ttl = f"#{rec.rank} {rec.prototype_id}\n{(rec.layout_type.value if rec.layout_type else '')} · 评分 {rec.quality_score if rec.quality_score is not None else float('nan'):.1f}" + (f" · 相似 {rec.similarity:.2f}" if rec.similarity is not None else "")
                if g is not None:
                    draw_topology(ax, g, set(g.nodes), title=ttl, seed=j)
                else:
                    ax.axis("off"); ax.set_title(ttl, fontsize=8)
            else:
                ax.axis("off")
        bb = f.add_subplot(gs[i, 3]).get_position(); f.axes[-1].remove()
        f.text(bb.x0, bb.y1 + 0.012, f"② 「{top}」内的 Top-3 可比原型（骨架；候选排除同商场楼层）", fontsize=s["fonts"]["size_annot"] + 0.5, fontweight="bold", va="bottom")
    f.legend(handles=legend_handles()[:1], loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.0), fontsize=s["fonts"]["size_annot"])
    f.suptitle("M01  阶段一流程：真实项目条件 → 布局类型期望评分 → 类型内高质量可比原型（三类城市各一例）", x=0.02, ha="left", fontweight="bold")
    _note(f, ctx, ctx.is_real_db)
    return savefig(f, out, "M01_stage1_pipeline_cases")


def m02_stage1_ranker_comparison(ctx: Ctx, out: Path) -> list[Path]:
    """Same 3 queries, Top-3 prototypes of four rankers: rule-weighted / extra-trees / LambdaMART / deep residual."""
    from mall_space_planner.reporting.graphdraw import draw_topology, legend_handles

    if ctx.db is None:
        return []
    s = load_style()
    rows = _pick_query_floors(ctx.db, 3)
    cfgs = [("规则加权（参照）", "configs/stage1/rule_knn.yaml"), ("极端随机树（本文默认）", "configs/stage1/extra_trees.yaml"), ("LambdaMART", "configs/stage1/lgbm_lambdarank.yaml"), ("Transformer+GNN 残差融合", "configs/stage1/deep_residual.yaml")]
    pipes = []
    for name, c in cfgs:
        try:
            pipes.append((name, ctx.pipeline_for(c)))
        except Exception as exc:  # noqa: BLE001
            ctx.notes.append(f"ranker {name} failed: {exc}")
    if not pipes:
        return []
    K = 3
    f, axes = plt.subplots(len(rows), len(pipes) * K, figsize=(s["figure"]["width_double"] * 1.3, 1.75 * len(rows) + 1.4))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        q = _query_from_row(ctx.db, r)
        mall = str(r[ctx.db.mall_id_col])
        for p, (name, pipe) in enumerate(pipes):
            recs = pipe.recommend(q, top_k=K, exclude_mall_ids={mall}, with_explanations=False, with_counterfactuals=False)
            for j in range(K):
                ax = axes[i, p * K + j]
                if j < len(recs):
                    rec = recs[j]
                    g = ctx.db.get_graph(rec.prototype_id)
                    ttl = f"#{rec.rank} · {(rec.layout_type.value if rec.layout_type else '')}\n评分 {rec.quality_score if rec.quality_score is not None else float('nan'):.1f}"
                    if g is not None:
                        draw_topology(ax, g, set(g.nodes), title=ttl, seed=j, title_pad=2)
                    else:
                        ax.axis("off"); ax.set_title(ttl, fontsize=7.5)
                else:
                    ax.axis("off")
        cl = int(r["city_cluster"]) if pd.notna(r.get("city_cluster")) else None
        axes[i, 0].text(-0.12, 0.5, f"案例 {i + 1}\n{label('clusters', str(cl)) if cl else ''}\n商场 {mall}\n建成 {r.get('layout_type', '')} · {float(r[ctx.db.label_col]):.1f}", transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"] - 0.5, rotation=90, linespacing=1.2)
    f.suptitle("M02  阶段一排序器比较：同一查询（真实条件）下不同排序器给出的 Top-3 可比原型（候选 = 同城市类别 / 同面积档 / 其他商场）", x=0.02, ha="left", fontweight="bold")
    f.legend(handles=legend_handles()[:1], loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    f.tight_layout(rect=(0.02, 0.03, 1, 0.90))
    for p, (name, _) in enumerate(pipes):
        b0, b1 = axes[0, p * K].get_position(), axes[0, p * K + K - 1].get_position()
        f.text((b0.x0 + b1.x1) / 2, b0.y1 + 0.045, name, ha="center", va="bottom", fontsize=s["fonts"]["size_label"], fontweight="bold", color=s["palette"]["ours"] if "本文" in name else "#222")
        f.add_artist(plt.Line2D([b0.x0, b1.x1], [b0.y1 + 0.04, b0.y1 + 0.04], color="#999", lw=0.8))
    _note(f, ctx, ctx.is_real_db)
    return savefig(f, out, "M02_stage1_ranker_comparison")


def m03_stage1_type_conditional_effect(ctx: Ctx, out: Path) -> list[Path]:
    """The same project conditions moved across the three city clusters: the type ranking changes (conditional effect),
    with one real built example of the first-ranked type per cluster."""
    from mall_space_planner.reporting.graphdraw import draw_topology

    if ctx.db is None:
        return []
    svc = ctx.service()
    s = load_style()
    df = ctx.db.cases
    clusters = sorted(df["city_cluster"].dropna().unique())[:3]
    base = _pick_query_floors(ctx.db, 1)[0]
    f, axes = plt.subplots(2, len(clusters), figsize=(s["figure"]["width_double"] * 1.1, 5.6), gridspec_kw={"height_ratios": [1.2, 1]})
    axes = np.atleast_2d(axes)
    for j, cl in enumerate(clusters):
        q = _query_from_row(ctx.db, base).model_copy(update={"city_cluster": int(cl)})
        types = svc.recommend_types(q)
        if not types:
            continue
        _draw_type_ranking(axes[0, j], types, f"{label('clusters', str(int(cl)))}：首选 {types.recommendations[0].layout_type}")
        top = types.recommendations[0].layout_type
        sub = df[(df["city_cluster"] == cl) & (df["layout_type"].astype(str) == top) & df[ctx.db.label_col].notna()]
        sub = sub[sub["has_graph"]] if "has_graph" in sub else sub
        ax = axes[1, j]
        if len(sub):
            r = sub.sort_values(ctx.db.label_col, ascending=False).iloc[0]
            g = ctx.db.get_graph(r[ctx.db.id_col])
            if g is not None:
                draw_topology(ax, g, set(g.nodes), title=f"该类城市中「{top}」的真实高分案例\n{r[ctx.db.id_col]} · 评分 {float(r[ctx.db.label_col]):.1f}", seed=j)
                continue
        ax.axis("off")
    f.suptitle("M03  类型条件效应：同一组项目条件放到三类城市，布局类型的期望评分排序随之改变（下排：各类城市首选类型的真实建成案例）", x=0.02, ha="left", fontweight="bold")
    f.text(0.02, 0.925, f"条件取自商场 {base[ctx.db.mall_id_col]}（{_conditions_text(ctx.db, base).replace(chr(10), '；')}）", fontsize=s["fonts"]["size_annot"] - 0.5, color="#555")
    f.tight_layout(rect=(0, 0, 1, 0.91))
    _note(f, ctx, ctx.is_real_db)
    return savefig(f, out, "M03_stage1_type_conditional_effect")


# ============================================================================================ Stage 2
def m04_stage2_method_comparison(ctx: Ctx, out: Path) -> list[Path]:
    if not ctx.samples:
        return []
    picks = _pick_cases(ctx.samples, per_type=1, types=["一字型", "简单环型", "多环型", "简单集中型", "复杂集中型"])
    ar = ctx.generator("ar")
    cols = [("骨架\n（阶段一原型）", None), ("规则扩展", "rule"), ("规则\n+ 16 次择优", "search16"), ("自回归图网络（本文）", "ar" if ar else None), ("自回归图网络（本文）\n+ 16 次择优", "ar16" if ar else None), ("真实建成拓扑", "gt")]
    cols = [(t, g) for t, g in cols if not (g is None and "骨架" not in t)]
    return _generation_grid(ctx, out, "M04_stage2_method_comparison", "M04", cols, picks, "阶段二方法比较：同一骨架 → 各方法生成的完整关键点网络 vs 真实建成（每行一种布局类型；骨架位置与比例尺固定）")


def m05_stage2_ablation_cases(ctx: Ctx, out: Path) -> list[Path]:
    if not ctx.samples or ctx.generator("ar") is None:
        return []
    picks = _pick_cases(ctx.samples, per_type=1, seed=11, types=["一字型", "简单环型", "多环型", "复杂集中型"])
    cols = [("骨架", None), ("完整模型（本文）\n标签顺序 · 集合似然", "ar")]
    for ttl, g in [("消融：单一接点标签", "ar_single"), ("消融：广度优先生长顺序", "ar_bfs"), ("消融：贪心生长顺序", "ar_greedy_order")]:
        if ctx.generator(g) is not None:
            cols.append((ttl, g))
    cols.append(("真实建成拓扑", "gt"))
    return _generation_grid(ctx, out, "M05_stage2_ablation_cases", "M05", cols, picks, "阶段二成分消融：同一骨架在完整模型与各消融变体下的生成结果（τ = 0.7 单次采样，同一随机种子）")


def m06_stage2_decoding_cases(ctx: Ctx, out: Path) -> list[Path]:
    if not ctx.samples or ctx.generator("ar") is None:
        return []
    picks = _pick_cases(ctx.samples, per_type=1, seed=5, types=["一字型", "多环型", "简单集中型"])
    cols = [("骨架", None), ("贪心解码（τ = 0）", "ar_greedy"), ("采样（τ = 0.7，默认）", "ar"), ("采样 + 16 次择优（本文推荐）", "ar16"), ("真实建成拓扑", "gt")]
    return _generation_grid(ctx, out, "M06_stage2_decoding_cases", "M06", cols, picks, "阶段二解码策略：贪心反复延长同一走廊、采样恢复分支多样性、择优用大纲指标拉回整体合理性（同一骨架）")


def m07_stage2_growth_steps(ctx: Ctx, out: Path, n_steps: int = 6) -> list[Path]:
    """Teacher-forced growth on 3 real samples of different types: each step = new node + 1–4 anchors (loop closing)."""
    from mall_space_planner.reporting.graphdraw import draw_topology, frame_of, full_layout, legend_handles, skeleton_layout
    from mall_space_planner.stage2.generators.ar_gnn import canonical_order
    from mall_space_planner.topology.convert import to_networkx

    if not ctx.samples:
        return []
    s = load_style()
    picks = _pick_cases(ctx.samples, per_type=1, seed=2, min_sk=6, max_sk=16, min_growth=n_steps + 2, types=["一字型", "简单环型", "复杂集中型"])
    if not picks:
        return []
    f, axes = plt.subplots(len(picks), n_steps + 2, figsize=(s["figure"]["width_double"] * 1.3, 1.75 * len(picks) + 1.0))
    axes = np.atleast_2d(axes)
    for r, smp in enumerate(picks):
        tg = to_networkx(smp.target)
        order = canonical_order(smp.skeleton, smp.target, "label")
        sk_nodes = set(smp.skeleton.nodes)
        sk_pos = skeleton_layout(to_networkx(smp.skeleton), seed=0)
        pos = full_layout(tg, sk_nodes, sk_pos, seed=0)
        steps = min(n_steps, len(order))
        shown = list(smp.skeleton.nodes) + list(order[:steps])
        frame = frame_of({k: pos[k] for k in shown})
        present = list(smp.skeleton.nodes)
        multi = 0
        for i in range(steps + 1):
            hl, ttl = None, "骨架（起点）"
            if i > 0:
                v = order[i - 1]; present.append(v)
                nbrs = [u for u in tg.neighbors(v) if u in present[:-1]]
                hl = [(u, v) for u in nbrs]
                multi += len(nbrs) > 1
                ttl = f"第 {i} 步：{len(nbrs)} 个接点" + ("（闭环）" if len(nbrs) > 1 else "")
            G = tg.subgraph(present).copy()
            draw_topology(axes[r, i], G, sk_nodes, pos={k: pos[k] for k in G.nodes}, title=ttl, highlight_edges=hl, seed=0, size_ref_n=len(shown), frame=frame, node_scale=1.15)
        n_new = tg.number_of_nodes() - len(sk_nodes)
        draw_topology(axes[r, steps + 1], tg, sk_nodes, pos=pos, title=f"…… 真实建成 {tg.number_of_nodes()} 单元\n新增 {n_new} 个，前 {steps} 步中 {multi} 步多接点", seed=0, size_ref_n=tg.number_of_nodes(), frame=frame_of(pos))
        for ax in axes[r, steps + 2:]:
            ax.axis("off")
        _row_label(axes[r, 0], smp)
    f.legend(handles=legend_handles() + [plt.Line2D([], [], color=s["palette"]["highlight"], lw=2.6, label="本步新建的连接（模型每步预测：接到哪里、是否再接第二 / 第三个接点形成环路）")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    f.suptitle("M07  自回归生长的训练目标：真实网络按语料顺序逐点生长，每一步的监督 = 新节点的接点集合（多接点 = 闭合回路，这是集合似然损失存在的原因）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0, 0.05, 1, 0.94))
    _note(f, ctx, ctx.is_real_corpus)
    return savefig(f, out, "M07_stage2_growth_steps")


# ============================================================================================ Stage 3
def _stage3_floors(ctx: Ctx, n: int = 3, max_nodes: int = 70, min_nodes: int = 14) -> list[str]:
    """Floors with *_M.csv + *_total.csv (+ skeleton), one per layout type when the corpus / DB tells the type."""
    gd = ctx.graph_dir
    if gd is None:
        return []
    ids = sorted({p.name[: -len("_M.csv")] for p in gd.glob("*_M.csv") if not p.name.endswith("_M_simplified.csv") and (gd / f"{p.name[: -len('_M.csv')]}_total.csv").exists() and (gd / f"{p.name[: -len('_M.csv')]}_M_simplified.csv").exists()})
    if not ids:
        return []
    type_of: dict[str, str] = {}
    for smp in ctx.samples:
        if smp.layout_type and smp.layout_type.value != "Unknown_Layout":
            type_of[smp.sample_id] = smp.layout_type.value
    if ctx.db is not None and "layout_type" in ctx.db.cases:
        for fid, lt in zip(ctx.db.cases[ctx.db.id_col].astype(str), ctx.db.cases["layout_type"]):
            if isinstance(lt, str) and fid not in type_of:
                type_of[fid] = lt
    from mall_space_planner.data.corpus_builder import load_target_csv

    rng = np.random.RandomState(9)
    test_ids = {smp.sample_id for smp in ctx.samples} if ctx.is_real_corpus else set(ids)
    pool = [i for i in ids if i in test_ids] or ids
    rng.shuffle(pool)
    chosen, used = [], set()
    for want_new_type in (True, False):
        for fid in pool:
            if fid in chosen or len(chosen) >= n:
                continue
            lt = type_of.get(fid)
            if want_new_type and (lt is None or lt in used):
                continue
            try:
                g = load_target_csv(gd / f"{fid}_M.csv")
            except Exception:  # noqa: BLE001
                continue
            if not (min_nodes <= g.num_nodes <= max_nodes) or not nx.is_connected(nx.Graph(g.edges())):
                continue
            chosen.append(fid); used.add(lt)
        if len(chosen) >= n:
            break
    return chosen[:n]


def _floor_type(ctx: Ctx, fid: str) -> str:
    for smp in ctx.samples:
        if smp.sample_id == fid and smp.layout_type:
            return smp.layout_type.value
    if ctx.db is not None and "layout_type" in ctx.db.cases:
        r = ctx.db.cases[ctx.db.cases[ctx.db.id_col].astype(str) == fid]
        if len(r) and isinstance(r.iloc[0]["layout_type"], str):
            return str(r.iloc[0]["layout_type"])
    return ""


def _draw_lines(ax, lines, **kw) -> None:  # noqa: ANN001
    for ln in lines:
        xy = np.asarray(ln.coords)
        ax.plot(xy[:, 0], xy[:, 1], **kw)


def _fit_stages(ctx: Ctx, fid: str, seed: int = 0):  # noqa: ANN201
    """Intermediate states of CorridorFitter.fit for one real floor: outline, inset+axis, init, final, plan, real."""
    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.data.legacy_adapter import load_graph_csv
    from mall_space_planner.geometry.planar_embed import PlanarEmbedParams, planar_corridor_embedding
    from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, render_corridors
    from mall_space_planner.stage3.fit import _init_on_inset, _roles, effective_depth
    from mall_space_planner.stage3.outline import m_positions_from_total_csv
    from mall_space_planner.stage3.skeleton import inset_region, medial_axis_graph
    from mall_space_planner.topology.convert import to_networkx

    gd = ctx.graph_dir
    topo = load_target_csv(gd / f"{fid}_M.csv")
    sk = load_graph_csv(gd / f"{fid}_M_simplified.csv", gd / f"{fid}_M_simplified_node_attributes.csv")
    sk_nodes = set(sk.nodes) & set(topo.nodes)
    outline = ctx.ds.outline(fid, 0)
    gt = m_positions_from_total_csv(gd / f"{fid}_total.csv", outline)
    p = FitParams(n_restarts=4, iters=100)
    fitter = CorridorFitter(p)
    depth = effective_depth(outline, p.shop_depth, p.depth_frac, p.depth_area_coef)
    inset = inset_region(outline, depth)
    _, lines = medial_axis_graph(outline, depth, px=p.raster_px)
    g = to_networkx(topo)
    _, info = planar_corridor_embedding(topo, PlanarEmbedParams(ortho_weight=0.0, relax_iters=0), skeleton_nodes=sk_nodes)
    roles = _roles(g, info)
    init = _init_on_inset(g, roles, list(info.get("outer_cycle", [])), inset, lines, depth, p.snap_dist, p.n_offsets)[0]
    res = fitter.fit(topo, outline, seed=seed, skeleton_nodes=sk_nodes)
    plan = render_corridors(topo, res.positions, outline, res.roles, RenderParams(), skeleton_nodes=sk_nodes)
    return dict(fid=fid, topo=topo, sk_nodes=sk_nodes, outline=outline, gt=gt, inset=inset, lines=lines, init=init, res=res, plan=plan, depth=depth)


def m08_stage3_fit_pipeline(ctx: Ctx, out: Path) -> list[Path]:
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan, poly

    fids = _stage3_floors(ctx, 3)
    if not fids:
        return []
    s = load_style()
    cols = ["① 轮廓（真实楼层）", "② 内缩区 + 中轴线", "③ 初始化：外环钉在内缩边界\n支路落在中轴", "④ 力松弛 + 交叉 / 越界修复\n+ 开角 + 角点吸附", "⑤ 一键成廊", "真实建成（同一网络的实测位置）"]
    f, axes = plt.subplots(len(fids), len(cols), figsize=(s["figure"]["width_double"] * 1.35, max(2.6 * len(fids) + 1.5, 5.0)))
    axes = np.atleast_2d(axes)
    for r, fid in enumerate(fids):
        st = _fit_stages(ctx, fid)
        o, ins = st["outline"], st["inset"]
        ax = axes[r, 0]; poly(ax, o.polygon, fc="#f4f4f4", ec="#333", lw=1.0); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        ax.set_title(f"{o.area / 1e4:.1f} 万 m² · 来源 {o.source}", fontsize=8)
        ax = axes[r, 1]; poly(ax, o.polygon, fc="#f4f4f4", ec="#333", lw=1.0); poly(ax, ins, fc="#e8eef7", ec="#4a6fa5", lw=0.8)
        _draw_lines(ax, st["lines"], color="#4a6fa5", lw=0.7, alpha=0.9); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        ax.set_title(f"内缩深度 {st['depth']:.0f} m（0.18·√面积）", fontsize=8)
        d0 = st["res"].diagnostics
        draw_network_in_outline(axes[r, 2], st["topo"], st["init"], o, st["sk_nodes"], f"{st['topo'].num_nodes} 节点 · 外环 {d0['roles']['outer']} · 支路 {d0['roles']['branch'] + d0['roles']['leaf']}")
        poly(axes[r, 2], ins, fc="none", ec="#4a6fa5", lw=0.6, ls="--")
        draw_network_in_outline(axes[r, 3], st["topo"], {k: np.asarray(v) for k, v in st["res"].positions.items()}, o, st["sk_nodes"], f"交叉 {d0['crossings']} · 越界边 {d0.get('edges_outside', 0)} · 锐角率 {d0['sharp_angle_rate']:.2f}\n墙向偏差 {d0['ortho_deviation_deg']:.0f}°")
        dp = st["plan"].diagnostics
        draw_plan(axes[r, 4], st["plan"], o, f"主廊 {dp['main_width_m']:.0f} m / 次廊 {dp['secondary_width_m']:.0f} m · {dp['n_entrances']} 出入口\n{dp['n_atria']} 中庭 · {dp['n_vertical_cores']} 竖向核 · 占比 {dp['corridor_ratio'] * 100:.0f}%")
        gt = st["gt"]
        if len(gt) >= 0.9 * st["topo"].num_nodes:
            draw_network_in_outline(axes[r, 5], st["topo"], gt, o, st["sk_nodes"], "黑 = 骨架（原型）节点", style="ink")
        else:
            axes[r, 5].axis("off")
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    f.suptitle("M08  阶段三拟合流程：从轮廓与关键点网络到走廊方案（3 个真实楼层；①–⑤ 为 CorridorFitter / render_corridors 的中间状态）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.01, 0.0, 1, 0.88))
    _col_headers(f, axes[0], cols, dy=0.06)
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M08_stage3_fit_pipeline")


def m09_stage3_rule_before_after(ctx: Ctx, out: Path) -> list[Path]:
    """Rules switched off vs on, same floor: (a) angle opening R0, (b) dead ends → entrances R9 (leaf-façade force),
    (c) edge containment (non-convex L outline)."""
    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.data.legacy_adapter import load_graph_csv
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan
    from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, outline_from_points, render_corridors
    from mall_space_planner.stage3.evaluate import evaluate_fit
    from mall_space_planner.stage3.fit import edges_outside

    fids = _stage3_floors(ctx, 3, max_nodes=50)
    if not fids:
        return []
    s = load_style()
    gd = ctx.graph_dir
    f, axes = plt.subplots(len(fids), 6, figsize=(s["figure"]["width_double"] * 1.35, max(2.6 * len(fids) + 1.5, 5.0)))
    axes = np.atleast_2d(axes)
    cols = ["(a) 无开角规则\n（min_angle = 0，w_angle = 0）", "(a) 开角规则 ≥ 60°（R0）", "(b) 无断头→立面力\n（w_leaf_facade = 0）", "(b) 断头 → 出入口（R9）", "(c) L 形轮廓·无边越界检查", "(c) L 形轮廓·边越界修复"]
    for r, fid in enumerate(fids):
        topo = load_target_csv(gd / f"{fid}_M.csv")
        sk = load_graph_csv(gd / f"{fid}_M_simplified.csv", gd / f"{fid}_M_simplified_node_attributes.csv")
        skn = set(sk.nodes) & set(topo.nodes)
        o = ctx.ds.outline(fid, 0)
        variants = [
            (FitParams(n_restarts=3, iters=80, min_angle_deg=0.0, w_angle=0.0), o, False),
            (FitParams(n_restarts=3, iters=80), o, False),
            (FitParams(n_restarts=3, iters=80, w_leaf_facade=0.0), o, True),
            (FitParams(n_restarts=3, iters=80), o, True),
        ]
        # (c) L outline with the same area
        A = o.area
        w = np.sqrt(A / 0.75)
        L = outline_from_points([(0, 0), (w, 0), (w, 0.5 * w), (0.5 * w, 0.5 * w), (0.5 * w, w), (0, w)])
        for c, (fp, oo, as_plan) in enumerate(variants):
            res = CorridorFitter(fp).fit(topo, oo, seed=0, skeleton_nodes=skn)
            P = {k: np.asarray(v) for k, v in res.positions.items()}
            if as_plan:
                plan = render_corridors(topo, res.positions, oo, res.roles, RenderParams(), skeleton_nodes=skn)
                ev = evaluate_fit(topo, res, plan, oo)
                draw_plan(axes[r, c], plan, oo, f"{plan.diagnostics['n_entrances']} 出入口 · {plan.diagnostics['n_vertical_cores']} 竖向核\n断头 {sum(1 for v in nx.Graph(topo.edges()).nodes if nx.Graph(topo.edges()).degree(v) == 1)} · 锐角率 {ev['sharp_angle_rate']:.2f}")
            else:
                d = res.diagnostics
                draw_network_in_outline(axes[r, c], topo, P, oo, skn, f"锐角率 {d['sharp_angle_rate']:.2f} · 墙向偏差 {d['ortho_deviation_deg']:.0f}°\n交叉 {d['crossings']}")
        # (c) edge containment: emulate "off" by fitting with a convex-hull site region, then counting outside edges
        fit_on = CorridorFitter(FitParams(n_restarts=3, iters=80))
        res_on = fit_on.fit(topo, L, seed=0, skeleton_nodes=skn)
        fit_off = _NoEdgeCheckFitter(FitParams(n_restarts=3, iters=80))
        res_off = fit_off.fit(topo, L, seed=0, skeleton_nodes=skn)
        for c, res in ((4, res_off), (5, res_on)):
            P = {k: np.asarray(v) for k, v in res.positions.items()}
            n_out = edges_outside(nx.Graph(topo.edges()), P, L.polygon.buffer(-1.0))
            draw_network_in_outline(axes[r, c], topo, P, L, skn, f"越界边 {n_out} · 交叉 {res.diagnostics['crossings']}")
            if n_out:
                g = nx.Graph(topo.edges())
                from mall_space_planner.stage3.fit import _EdgeInsideChecker

                ec = _EdgeInsideChecker(L.polygon.buffer(-1.0))
                for u, v in ec.bad_edges(g, P):
                    axes[r, c].plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color="#C0392B", lw=2.4, zorder=7)
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}\n{topo.num_nodes} 节点", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    f.suptitle("M09  阶段三规则的前后对比（同一楼层、同一网络、同一随机种子）：(a) 开角规则 (b) 断头→出入口 (c) 非凸轮廓的边越界修复（红 = 越出建筑的走廊段）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.01, 0.0, 1, 0.88))
    _col_headers(f, axes[0], cols, dy=0.06)
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M09_stage3_rule_before_after")


class _NoEdgeCheckFitter:
    """``CorridorFitter`` with the edge-containment checks disabled (the pre-fix behaviour), for the before/after figure.
    Implemented by giving the checker the convex hull of the site so no edge is ever flagged."""

    def __init__(self, params) -> None:  # noqa: ANN001
        from mall_space_planner.stage3 import CorridorFitter

        self.inner = CorridorFitter(params)

    def fit(self, topo, outline, **kw):  # noqa: ANN001, ANN201
        from mall_space_planner.stage3.fit import _EdgeInsideChecker

        inner = self.inner
        hull = outline.polygon.convex_hull.buffer(5.0)

        def _ec(inset):  # noqa: ANN001, ANN202
            return _EdgeInsideChecker(hull)

        inner._edge_checker = _ec  # type: ignore[method-assign]
        orig_repair = inner.repair_outside
        inner.repair_outside = lambda g, pos, *a, **k: pos  # type: ignore[method-assign]
        try:
            return inner.fit(topo, outline, **kw)
        finally:
            inner.repair_outside = orig_repair  # type: ignore[method-assign]


def m10_stage3_transfer_outlines(ctx: Ctx, out: Path) -> list[Path]:
    """Same real network into: its own outline / the closest-area outline of another mall / a hand-drawn L / U."""
    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.data.legacy_adapter import load_graph_csv
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan
    from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, outline_from_points, render_corridors

    fids = _stage3_floors(ctx, 3, max_nodes=60)
    if not fids:
        return []
    s = load_style()
    gd = ctx.graph_dir
    f, axes = plt.subplots(len(fids), 5, figsize=(s["figure"]["width_double"] * 1.25, max(2.7 * len(fids) + 1.4, 5.0)))
    axes = np.atleast_2d(axes)
    cols = ["网络（真实建成位置）", "自身轮廓 → 成廊", "另一真实商场的轮廓（面积最近）", "手绘 L 形轮廓（等面积）", "手绘 U 形轮廓（等面积）"]
    fitter = CorridorFitter(FitParams(n_restarts=4, iters=100))
    for r, fid in enumerate(fids):
        topo = load_target_csv(gd / f"{fid}_M.csv")
        sk = load_graph_csv(gd / f"{fid}_M_simplified.csv", gd / f"{fid}_M_simplified_node_attributes.csv")
        skn = set(sk.nodes) & set(topo.nodes)
        o = ctx.ds.outline(fid, 0)
        from mall_space_planner.stage3.outline import m_positions_from_total_csv

        gt = m_positions_from_total_csv(gd / f"{fid}_total.csv", o)
        draw_network_in_outline(axes[r, 0], topo, gt, o, skn, f"{topo.num_nodes} 节点 · {o.area / 1e4:.1f} 万 m²", style="ink")
        A = o.area
        w = np.sqrt(A / 0.75)
        L = outline_from_points([(0, 0), (w, 0), (w, 0.5 * w), (0.5 * w, 0.5 * w), (0.5 * w, w), (0, w)])
        wu = np.sqrt(A / 0.64)
        U = outline_from_points([(0, 0), (wu, 0), (wu, 0.8 * wu), (0.7 * wu, 0.8 * wu), (0.7 * wu, 0.3 * wu), (0.3 * wu, 0.3 * wu), (0.3 * wu, 0.8 * wu), (0, 0.8 * wu)])
        other = None
        mall = fid.rsplit("_", 1)[0]
        try:
            for cand in ctx.ds.similar_outlines(A, k=6, exclude_mall=mall):
                if cand.rsplit("_", 1)[0] != mall:
                    cf = cand.rsplit("_", 1)[0]
                    other = (cand, ctx.ds.outline(cf, int(cand.rsplit("_", 1)[1]) if cand.count("_") >= 2 else 0))
                    break
        except Exception:  # noqa: BLE001
            other = None
        if other is None:  # sandbox: no other floor → a rectangle of the same area
            other = ("矩形（示意）", outline_from_points([(0, 0), (np.sqrt(A * 2.2), 0), (np.sqrt(A * 2.2), np.sqrt(A / 2.2)), (0, np.sqrt(A / 2.2))]))
        for c, (name, oo) in enumerate([("自身", o), (other[0], other[1]), ("L", L), ("U", U)], start=1):
            res = fitter.fit(topo, oo, seed=0, skeleton_nodes=skn)
            plan = render_corridors(topo, res.positions, oo, res.roles, RenderParams(), skeleton_nodes=skn)
            d, dp = res.diagnostics, plan.diagnostics
            draw_plan(axes[r, c], plan, oo, f"{name if c == 2 else ''}{' · ' if c == 2 else ''}交叉 {d['crossings']} · 越界边 {d.get('edges_outside', 0)} · 锐角率 {d['sharp_angle_rate']:.2f}\n{dp['n_entrances']} 出入口 · {dp['n_atria']} 中庭 · 占比 {dp['corridor_ratio'] * 100:.0f}%")
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    f.suptitle("M10  轮廓自适应：同一个真实关键点网络放入自身轮廓、另一真实商场轮廓与手绘 L / U 形轮廓（骨架环 = 外环、走廊不越出建筑、无交叉）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.01, 0.0, 1, 0.88))
    _col_headers(f, axes[0], cols, dy=0.06)
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M10_stage3_transfer_outlines")


def m11_renovation_versions(ctx: Ctx, out: Path) -> list[Path]:
    """Renovation v1 (all skeleton nodes anchored) vs v2 (entrances only + Stage-1 score) vs v2b (cycle-neutral
    objective) on 3 real floors; indicator deltas under each panel."""
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan
    from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams
    from mall_space_planner.stage3 import renovate as rv
    from mall_space_planner.stage3.renovate import renovate_floor

    fids = _stage3_floors(ctx, 3, max_nodes=70, min_nodes=14)
    if not fids:
        return []
    s = load_style()
    gen = ctx.generator("ar16") or ctx.generator("search16")
    gen_name = "AR-GNN + 16 次择优" if ctx.generator("ar16") else "规则 + 16 次择优"
    fitter = CorridorFitter(FitParams(n_restarts=4, iters=100))
    score_fn = None
    if ctx.db is not None:
        try:
            svc = ctx.service()
            cases = ctx.db.cases
            r0 = cases[cases[ctx.db.id_col].astype(str) == fids[0]]
            cond = _query_from_row(ctx.db, r0.iloc[0]) if len(r0) else _query_from_row(ctx.db, cases.iloc[0])
            score_fn = rv.make_score_fn(svc, cond)
        except Exception as exc:  # noqa: BLE001
            ctx.notes.append(f"score_fn unavailable: {exc}")

    def _v2_objective(before_ind, cand, sf=None):  # noqa: ANN001, ANN202
        ind = rv.topo_indicators(cand)
        dead = max(0, ind["n_dead_ends"] - before_ind["n_dead_ends"])
        loops = max(0.0, (before_ind["num_cycles"] - ind["num_cycles"]) / max(before_ind["num_cycles"], 1))
        aspl = max(0.0, (ind["avg_shortest_path"] - before_ind["avg_shortest_path"]) / max(before_ind["avg_shortest_path"], 1e-9))
        obj = 1.0 * dead + 3.0 * loops + 3.0 * aspl + 2.0 * max(0, ind["n_components"] - 1)
        if sf is not None:
            try:
                obj -= 4.0 * float(sf(cand))
            except Exception:  # noqa: BLE001
                pass
        return obj

    versions = [("v1：锚定全部骨架节点\n目标含「回路越多越好」", dict(keep_skeleton_positions=True, score_fn=None), _v2_objective),
                ("v2：仅锚定既有出入口 + 阶段一预测评分\n（骨架位置重排）", dict(keep_skeleton_positions=False, score_fn=score_fn), _v2_objective),
                ("v2b（本文）：回路中性 + 整合度下降惩罚", dict(keep_skeleton_positions=False, score_fn=score_fn), None)]
    f, axes = plt.subplots(len(fids), 1 + 2 * len(versions), figsize=(s["figure"]["width_double"] * 1.55, max(2.9 * len(fids) + 1.6, 5.2)), gridspec_kw={"width_ratios": [1] + [1, 1] * len(versions)})
    axes = np.atleast_2d(axes)
    orig_obj = rv.renovation_objective
    for r, fid in enumerate(fids):
        first = None
        for c, (vname, kw, obj) in enumerate(versions):
            rv.renovation_objective = obj or orig_obj
            try:
                res = renovate_floor(fid, ctx.graph_dir, ctx.ds, gen, fitter, RenderParams(), seed=0, n_candidates=6, **kw)
            finally:
                rv.renovation_objective = orig_obj
            if first is None:
                first = res
                ib = res["ind_b"]
                draw_network_in_outline(axes[r, 0], res["before"], res["gt"], res["outline"], res["sk_nodes"], f"现状：{ib['num_nodes']} 节点 · {ib['num_cycles']} 回路 · 断头 {ib['n_dead_ends']}\nASPL {ib['avg_shortest_path']:.2f} · 整合度 {ib['closeness_mean']:.3f}", style="ink")
            ia, ib = res["ind_a"], res["ind_b"]
            P = {k: np.asarray(v) for k, v in res["res"].positions.items()}

            def d(k, fmt="{:+.2f}"):  # noqa: ANN001, ANN202
                return fmt.format(ia[k] - ib[k])

            draw_network_in_outline(axes[r, 1 + 2 * c], res["after"], P, res["outline"], res["sk_nodes"], f"{ia['num_cycles']} 回路 ({d('num_cycles', '{:+d}')}) · 断头 {ia['n_dead_ends']} ({d('n_dead_ends', '{:+d}')})\nASPL {ia['avg_shortest_path']:.2f} ({d('avg_shortest_path')}) · 整合度 {d('closeness_mean', '{:+.3f}')}", anchored=res.get("anchored", ()))
            dp = res["plan"].diagnostics
            draw_plan(axes[r, 2 + 2 * c], res["plan"], res["outline"], f"交叉 {res['row']['after_crossings']} · {dp['n_entrances']} 出入口 · {dp['n_atria']} 中庭\n锐角率 {res['row']['after_sharp_angle_rate']:.2f} · 占比 {dp['corridor_ratio'] * 100:.0f}%")
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    f.suptitle(f"M11  改造工作流三个版本的对比（同一楼层、同一轮廓、{gen_name}、同一随机种子；红框 = 锚定节点；括号 = 相对现状的变化）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.01, 0.0, 1, 0.86))
    hdr = ["现状（真实建成）"]
    for vname, _, _ in versions:
        hdr += [vname, "走廊方案"]
    _col_headers(f, axes[0], hdr, dy=0.09)
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M11_renovation_versions")


def m12_data_levels(ctx: Ctx, out: Path) -> list[Path]:
    """Three real floors of different types: colour-block plan → complete M network (ink) → skeleton only."""
    from mall_space_planner.hub.plates import draw_plate, load_floor
    from mall_space_planner.hub.viz import draw_network_in_outline
    from mall_space_planner.schemas import TopologyGraph

    fids = _stage3_floors(ctx, 3, max_nodes=120)
    if not fids:
        return []
    s = load_style()
    f, axes = plt.subplots(len(fids), 3, figsize=(s["figure"]["width_double"], 2.5 * len(fids) + 1.0))
    axes = np.atleast_2d(axes)
    for r, fid in enumerate(fids):
        rn = load_floor(fid, ctx.ds, ctx.graph_dir)
        used = draw_plate(axes[r, 0], rn, with_plan=True, title="色块平面图 + 完整关键点网络 M" if rn["plan_png"] else "轮廓 + 完整关键点网络 M（无平面图）")
        full, sk, pos, o = rn["full"], rn["skeleton"], rn["positions"], rn["outline"]
        draw_network_in_outline(axes[r, 1], full, pos, o, set(sk.nodes) if sk else None, f"完整网络 M：{full.num_nodes} 节点 / {full.num_edges} 边", style="ink")
        if sk is not None:
            sk_only = TopologyGraph(adjacency={u: [v for v in sk.adjacency[u] if v in pos] for u in sk.nodes if u in pos})
            draw_network_in_outline(axes[r, 2], sk_only, pos, o, set(sk.nodes), f"骨架 M_s（阶段一原型）：{sk.num_nodes} 节点 / {sk.num_edges} 边", style="ink")
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    f.suptitle("M12  同一楼层的三个数据层级：色块平面（阶段三图像数据）→ 完整走廊关键点网络 M（阶段二目标 / 阶段三输入）→ 骨架 M_s（阶段一原型 / 阶段二输入）", x=0.02, ha="left", fontweight="bold", fontsize=s["fonts"]["size_title"] - 1)
    f.tight_layout(rect=(0.01, 0.0, 1, 0.93))
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M12_data_levels")


ALL = [m01_stage1_pipeline_cases, m02_stage1_ranker_comparison, m03_stage1_type_conditional_effect, m04_stage2_method_comparison, m05_stage2_ablation_cases, m06_stage2_decoding_cases, m07_stage2_growth_steps, m08_stage3_fit_pipeline, m09_stage3_rule_before_after, m10_stage3_transfer_outlines, m11_renovation_versions, m12_data_levels]

__all__ = ["ALL", "Ctx", "build_ctx", *[fn.__name__ for fn in ALL]]
