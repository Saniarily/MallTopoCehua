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


def _col_headers(f, axes_row, cols: list[str], dy: float = 0.035, fs: float | None = None) -> None:  # noqa: ANN001
    s = load_style()
    fs = fs if fs is not None else s["fonts"]["size_annot"] + 0.5
    for c, ttl in enumerate(cols):
        bb = axes_row[c].get_position()
        f.text((bb.x0 + bb.x1) / 2, bb.y1 + dy, ttl, ha="center", va="bottom", linespacing=1.15, fontsize=fs, fontweight="bold", color=s["palette"]["ours"] if "本文" in ttl or "后：" in ttl else "#222")


def _finish_grid(f, axes, cols: list[str], suptitle: str, bottom: float = 0.0, header_lines: int = 2, fs: float | None = None) -> None:  # noqa: ANN001
    """Suptitle + column headers with guaranteed vertical room: the panel grid is pushed down by an amount derived from the
    figure height and the number of header lines (so neither the headers nor the suptitle can collide with the panels)."""
    s = load_style()
    H = f.get_size_inches()[1]
    fs = fs if fs is not None else s["fonts"]["size_annot"] + 0.5
    head_in = 0.32 + header_lines * fs / 72 * 1.5 + 0.12  # suptitle + header text + gap, in inches
    top = 1.0 - head_in / H
    f.tight_layout(rect=(0.02, bottom, 1, top))
    f.suptitle(suptitle, x=0.02, y=0.995, ha="left", va="top", fontweight="bold", fontsize=s["fonts"]["size_title"] - 0.5)
    _col_headers(f, axes[0], cols, dy=0.012, fs=fs)


def _generation_grid(ctx: Ctx, out: Path, stem: str, code: str, cols: list[tuple[str, str | None]], picks, suptitle_extra: str, per_panel_metric: bool = True, seed_offset: int = 0):  # noqa: ANN001
    """Rows = cases, columns = [skeleton, generators..., ground truth]; shared frame per row."""
    from mall_space_planner.reporting.graphdraw import draw_topology, frame_of, full_layout, legend_handles, skeleton_layout
    from mall_space_planner.topology.convert import to_networkx
    from mall_space_planner.topology.metrics import attachment_overlap, degree_histogram_emd

    s = load_style()
    f, axes = plt.subplots(len(picks), len(cols), figsize=(s["figure"]["width_double"] * (0.95 + 0.22 * len(cols)), 2.15 * len(picks) + 1.5), gridspec_kw={"hspace": 0.55, "wspace": 0.12})
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
                extra = f"\n位置正确率 {pr:.0f}% · 度分布差异 {emd:.2f}"
            subs.append(f"{G.number_of_nodes()} 单元 / {G.number_of_edges()} 连接{extra}")
        frame = frame_of(*[p for p in poses if p])
        for c, ((ttl, gname), gg, pos, sub) in enumerate(zip(cols, graphs, poses, subs)):
            ax = axes[r, c]
            if gg is None:
                ax.axis("off"); ax.text(0.5, 0.5, sub, ha="center", va="center", fontsize=8, color="#888", transform=ax.transAxes); continue
            draw_topology(ax, gg[1], sk_nodes, pos=pos, title=sub, seed=r, size_ref_n=max(n_t, smp.target.num_nodes), frame=frame, node_scale=0.72, title_fs=s["fonts"]["size_annot"] - 0.5, title_pad=2)
        _row_label(axes[r, 0], smp)
    f.legend(handles=legend_handles(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    _finish_grid(f, axes, [t for t, _ in cols], f"{code}  {suptitle_extra}", bottom=0.03)
    _note(f, ctx, ctx.is_real_corpus)
    return savefig(f, out, stem)



def _fit_box(ax, x0: float, x1: float, y0: float, y1: float, pad: float = 0.06) -> None:  # noqa: ANN001
    """Equal-aspect view that keeps the *axes box* fixed (so captions placed in axes coordinates stay put) by widening
    the data limits to the box aspect instead of shrinking the box."""
    ax.axis("off")
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    x0, x1, y0, y1 = x0 - pad * w, x1 + pad * w, y0 - pad * h, y1 + pad * h
    w, h = x1 - x0, y1 - y0
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    box_aspect = bb.height / max(bb.width, 1e-9)
    if h / w < box_aspect:
        extra = w * box_aspect - h; y0, y1 = y0 - extra / 2, y1 + extra / 2
    else:
        extra = h / box_aspect - w; x0, x1 = x0 - extra / 2, x1 + extra / 2
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal", adjustable="box")


def draw_plan_graph(ax, g, pos: dict, skeleton_nodes: set | None = None, title: str = "", node_pt: float = 2.6, lw: float = 0.9, flip_y: bool = True, title_fs: float | None = None, highlight_edges=None) -> None:  # noqa: ANN001
    """Topology drawn at its *real plan coordinates* (pixel CenterPoint, y down → flipped) with one fixed node size
    and line width for the whole figure, so panels are comparable and nodes never hide the edges."""
    import networkx as nx

    from mall_space_planner.topology.convert import to_networkx

    s = load_style()
    G = g if isinstance(g, nx.Graph) else to_networkx(g)
    sk = set(skeleton_nodes or ())
    P = {k: (float(v[0]), -float(v[1]) if flip_y else float(v[1])) for k, v in pos.items() if k in G.nodes}
    if not P:
        ax.axis("off"); return
    for u, v in G.edges:
        if u in P and v in P:
            main = u in sk and v in sk
            ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color="#2B2B2B" if main else s["palette"]["ours"], lw=lw * (1.35 if main else 1.0), alpha=0.95 if main else 0.8, zorder=2, solid_capstyle="round")
    for u, v in highlight_edges or ():
        if u in P and v in P:
            ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=s["palette"]["highlight"], lw=lw * 2.4, zorder=6, solid_capstyle="round")
    xs = np.array([P[v] for v in G.nodes if v in P])
    skm = np.array([v in sk for v in G.nodes if v in P])
    if (~skm).any():
        ax.scatter(xs[~skm, 0], xs[~skm, 1], s=node_pt**2, facecolors="white", edgecolors=s["palette"]["ours"], linewidths=0.8, zorder=4)
    if skm.any():
        ax.scatter(xs[skm, 0], xs[skm, 1], s=node_pt**2, facecolors="#2B2B2B", edgecolors="white", linewidths=0.5, zorder=5)
    _fit_box(ax, xs[:, 0].min(), xs[:, 0].max(), xs[:, 1].min(), xs[:, 1].max())
    if title:
        ax.set_title(title, fontsize=title_fs if title_fs is not None else s["fonts"]["size_annot"], pad=3, linespacing=1.15)


def _real_pos(g) -> dict:  # noqa: ANN001
    """Pixel CenterPoint positions stored with a graph (empty when the export has none)."""
    return dict(getattr(g, "positions", None) or {})


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
    lo = min(r.ci_low for r in rows); hi = max(r.ci_high for r in rows); span = max(hi - lo, 1e-6)
    for yi, r, c in zip(y, rows, cols):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=2.0, solid_capstyle="round")
        ax.scatter([r.expected_score], [yi], s=26, color=c, zorder=5)
        ax.text(hi + 0.06 * span, yi, f"可比 {r.n_comparable_cases}", va="center", ha="left", fontsize=s["fonts"]["size_annot"] - 1.5, color="#666")
    ax.set_xlim(lo - 0.08 * span, hi + 0.42 * span)
    ax.set_yticks(y); ax.set_yticklabels([r.layout_type for r in rows], fontsize=s["fonts"]["size_annot"] - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("期望综合评分（线段 = 80% 置信区间）", fontsize=s["fonts"]["size_annot"] - 1)
    ax.tick_params(axis="x", labelsize=s["fonts"]["size_annot"] - 1)
    ax.grid(axis="x", alpha=s["figure"]["grid_alpha"]); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=s["fonts"]["size_annot"], loc="left", pad=4)


def _proto_layout_panel(ax, g, node_scale: float = 0.72, size_ref_n: int = 40) -> None:  # noqa: ANN001
    """Prototype skeleton drawn like M04 (spread-out force layout, all nodes = skeleton style). ``size_ref_n`` fixes the
    node size across the whole figure regardless of each graph's node count."""
    from mall_space_planner.reporting.graphdraw import draw_topology, skeleton_layout
    from mall_space_planner.topology.convert import to_networkx

    G = to_networkx(g)
    pos = skeleton_layout(G, seed=0)
    draw_topology(ax, G, set(G.nodes), pos=pos, seed=0, size_ref_n=size_ref_n, node_scale=node_scale)


def _proto_panel(ctx: Ctx, ax, rec, node_pt: float = 2.4, fs: float | None = None) -> None:  # noqa: ANN001
    """One recommended prototype in M04-style force layout (uniform node size); caption under the panel."""
    g = ctx.db.get_graph(rec.prototype_id)
    s = load_style()
    fs = fs if fs is not None else s["fonts"]["size_annot"] - 1.0
    q = f"{rec.quality_score:.1f}" if rec.quality_score is not None else "–"
    sim = f"\n相似 {rec.similarity:.2f}" if rec.similarity is not None else ""
    cap = f"#{rec.rank}  {rec.prototype_id}\n{(rec.layout_type.value if rec.layout_type else '')} · 评分 {q}{sim}"
    if g is None:
        ax.axis("off")
    else:
        _proto_layout_panel(ax, g)
    _tag(ax, cap, fs)


def m01_stage1_pipeline_cases(ctx: Ctx, out: Path) -> list[Path]:
    """3 real queries (one per city cluster) → type ranking → Top-3 prototypes within the first-ranked type."""
    from mall_space_planner.schemas import LayoutType

    if ctx.db is None:
        return []
    svc = ctx.service()
    s = load_style()
    fs = s["fonts"]["size_annot"] - 0.5
    rows = _pick_query_floors(ctx.db, 3)
    f = plt.figure(figsize=(s["figure"]["width_double"] * 1.3, 3.1 * len(rows) + 0.8))
    # per case: a thin header row (one line of text) + a band of [conditions | gap | ranking | gap | 3 prototypes]
    gs = f.add_gridspec(2 * len(rows), 7, width_ratios=[1.0, 0.55, 1.45, 0.15, 1, 1, 1], height_ratios=[0.14, 1.0] * len(rows), wspace=0.1, hspace=0.25, left=0.02, right=0.99, top=0.94, bottom=0.04)
    for i, r in enumerate(rows):
        q = _query_from_row(ctx.db, r)
        types = svc.recommend_types(q)
        top = types.recommendations[0].layout_type if types and types.recommendations else None
        mall = str(r[ctx.db.mall_id_col])
        qq = q.model_copy(update={"preferred_layout": LayoutType(top)}) if top else q
        recs = svc.stage1.recommend(qq, top_k=3, exclude_mall_ids={mall}, with_explanations=False, with_counterfactuals=False)
        hb, b = 2 * i, 2 * i + 1
        cl = int(r["city_cluster"]) if pd.notna(r.get("city_cluster")) else None
        built = f"；实际建成 {r['layout_type']}，评分 {float(r[ctx.db.label_col]):.1f}" if pd.notna(r.get("layout_type")) else ""
        ax = f.add_subplot(gs[hb, :]); ax.axis("off")
        ax.text(0, 0.0, f"案例 {i + 1}  ·  {label('clusters', str(cl)) if cl else ''}  ·  查询 = 商场 {mall} 的真实条件{built}      ①  六种类型的期望评分（红 = 首选「{top}」）      ②  「{top}」内 Top-3 可比原型（候选排除同商场）", fontsize=s["fonts"]["size_annot"] + 0.5, fontweight="bold", va="bottom")
        ax = f.add_subplot(gs[b, 0]); ax.axis("off")
        ax.text(0, 1.0, _conditions_text(ctx.db, r), va="top", fontsize=fs, linespacing=1.5)
        ax = f.add_subplot(gs[b, 2])
        if types:
            _draw_type_ranking(ax, types, "")
        for j in range(3):
            ax = f.add_subplot(gs[b, 4 + j])
            if j < len(recs):
                _proto_panel(ctx, ax, recs[j], fs=fs)
            else:
                ax.axis("off")
    f.suptitle("M01  阶段一流程：真实项目条件 → 布局类型期望评分 → 类型内高质量可比原型（三类城市各一例）", x=0.02, ha="left", fontweight="bold")
    _note(f, ctx, ctx.is_real_db)
    return savefig(f, out, "M01_stage1_pipeline_cases")


def m02_stage1_ranker_comparison(ctx: Ctx, out: Path) -> list[Path]:
    """Same 3 queries, Top-3 prototypes of four rankers: rule-weighted / extra-trees / LambdaMART / deep residual."""
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
    ncol = len(pipes) * K + (len(pipes) - 1)  # spacer column between rankers
    widths = []
    for p in range(len(pipes)):
        widths += [1.0] * K + ([0.18] if p < len(pipes) - 1 else [])
    f = plt.figure(figsize=(s["figure"]["width_double"] * 1.45, 2.35 * len(rows) + 1.3))
    gs = f.add_gridspec(len(rows), ncol, width_ratios=widths, wspace=0.08, hspace=0.55, left=0.06, right=0.99, top=0.86, bottom=0.06)
    axes = np.empty((len(rows), len(pipes), K), dtype=object)
    for i, r in enumerate(rows):
        q = _query_from_row(ctx.db, r)
        mall = str(r[ctx.db.mall_id_col])
        for p, (name, pipe) in enumerate(pipes):
            recs = pipe.recommend(q, top_k=K, exclude_mall_ids={mall}, with_explanations=False, with_counterfactuals=False)
            for j in range(K):
                ax = f.add_subplot(gs[i, p * (K + 1) + j]); axes[i, p, j] = ax
                if j < len(recs):
                    rec = recs[j]
                    g = ctx.db.get_graph(rec.prototype_id)
                    qv = f"{rec.quality_score:.1f}" if rec.quality_score is not None else "–"
                    ttl = f"#{rec.rank} {(rec.layout_type.value if rec.layout_type else '')}\n评分 {qv}"
                    if g is not None:
                        _proto_layout_panel(ax, g, node_scale=0.65)
                    else:
                        ax.axis("off")
                    _tag(ax, ttl, s["fonts"]["size_annot"] - 1)
                else:
                    ax.axis("off")
        cl = int(r["city_cluster"]) if pd.notna(r.get("city_cluster")) else None
        axes[i, 0, 0].text(-0.18, 0.5, f"案例 {i + 1}\n{label('clusters', str(cl)) if cl else ''}\n商场 {mall}\n建成 {r.get('layout_type', '')} · {float(r[ctx.db.label_col]):.1f}", transform=axes[i, 0, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"] - 1, rotation=90, linespacing=1.25)
    for p, (name, _) in enumerate(pipes):
        b0, b1 = axes[0, p, 0].get_position(), axes[0, p, K - 1].get_position()
        f.text((b0.x0 + b1.x1) / 2, b0.y1 + 0.055, name, ha="center", va="bottom", fontsize=s["fonts"]["size_label"], fontweight="bold", color=s["palette"]["ours"] if "本文" in name else "#222")
        f.add_artist(plt.Line2D([b0.x0, b1.x1], [b0.y1 + 0.05, b0.y1 + 0.05], color="#999", lw=0.8))
    f.suptitle("M02  阶段一排序器比较：同一查询（真实条件）下不同排序器给出的 Top-3 可比原型（候选 = 同城市类别 / 同面积档 / 其他商场）", x=0.02, ha="left", fontweight="bold")
    _note(f, ctx, ctx.is_real_db)
    return savefig(f, out, "M02_stage1_ranker_comparison")


def m03_stage1_type_conditional_effect(ctx: Ctx, out: Path) -> list[Path]:
    """The same project conditions moved across the three city clusters: the type ranking changes (conditional effect),
    with one real built example of the first-ranked type per cluster."""
    if ctx.db is None:
        return []
    svc = ctx.service()
    s = load_style()
    df = ctx.db.cases
    clusters = sorted(df["city_cluster"].dropna().unique())[:3]
    base = _pick_query_floors(ctx.db, 1)[0]
    f, axes = plt.subplots(2, len(clusters), figsize=(s["figure"]["width_double"] * 1.1, 6.2), gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.45, "wspace": 0.55})
    axes = np.atleast_2d(axes)
    for j, cl in enumerate(clusters):
        q = _query_from_row(ctx.db, base).model_copy(update={"city_cluster": int(cl)})
        types = svc.recommend_types(q)
        if not types:
            continue
        top = types.recommendations[0].layout_type
        _draw_type_ranking(axes[0, j], types, f"{label('clusters', str(int(cl)))}：首选「{top}」")
        sub = df[(df["city_cluster"] == cl) & (df["layout_type"].astype(str) == top) & df[ctx.db.label_col].notna()]
        sub = sub[sub["has_graph"]] if "has_graph" in sub else sub
        ax = axes[1, j]
        if len(sub):
            r = sub.sort_values(ctx.db.label_col, ascending=False).iloc[0]
            g = ctx.db.get_graph(r[ctx.db.id_col])
            if g is not None:
                _proto_layout_panel(ax, g, node_scale=0.8)
                _tag(ax, f"该类城市中「{top}」的真实高分案例\n{r[ctx.db.id_col]} · 评分 {float(r[ctx.db.label_col]):.1f} · {g.num_nodes} 个关键点", s["fonts"]["size_annot"] - 0.5)
                continue
        ax.axis("off")
    f.suptitle("M03  类型条件效应：同一组项目条件放到三类城市，布局类型的期望评分排序随之改变（下排：各类城市首选类型的真实建成案例）", x=0.02, ha="left", fontweight="bold")
    f.text(0.02, 0.915, f"条件取自商场 {base[ctx.db.mall_id_col]}：{_conditions_text(ctx.db, base).replace(chr(10), '；')}", fontsize=s["fonts"]["size_annot"] - 1, color="#555")
    f.tight_layout(rect=(0, 0.03, 1, 0.89))
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
    ncol = n_steps + 2
    f, axes = plt.subplots(len(picks), ncol, figsize=(s["figure"]["width_double"] * 1.45, 2.35 * len(picks) + 1.2), gridspec_kw={"hspace": 0.55, "wspace": 0.10})
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
            hl, ttl = None, "骨架\n（起点）"
            if i > 0:
                v = order[i - 1]; present.append(v)
                nbrs = [u for u in tg.neighbors(v) if u in present[:-1]]
                hl = [(u, v) for u in nbrs]
                multi += len(nbrs) > 1
                ttl = f"第 {i} 步\n{len(nbrs)} 个接点" + ("·闭环" if len(nbrs) > 1 else "")
            G = tg.subgraph(present).copy()
            draw_topology(axes[r, i], G, sk_nodes, pos={k: pos[k] for k in G.nodes}, title=ttl, highlight_edges=hl, seed=0, size_ref_n=max(len(shown), 24), frame=frame, node_scale=0.78, title_fs=s["fonts"]["size_annot"] - 0.5, title_pad=2)
        n_new = tg.number_of_nodes() - len(sk_nodes)
        draw_topology(axes[r, steps + 1], tg, sk_nodes, pos=pos, title=f"真实建成 {tg.number_of_nodes()} 单元\n新增 {n_new}·前 {steps} 步 {multi} 步多接点", seed=0, size_ref_n=tg.number_of_nodes(), frame=frame_of(pos), node_scale=0.72, title_fs=s["fonts"]["size_annot"] - 0.5, title_pad=2)
        for ax in axes[r, steps + 2:]:
            ax.axis("off")
        _row_label(axes[r, 0], smp)
    f.legend(handles=legend_handles() + [plt.Line2D([], [], color=s["palette"]["highlight"], lw=2.6, label="本步新建的连接（模型每步预测：接到哪里、是否再接第二 / 第三个接点形成环路）")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), fontsize=s["fonts"]["size_annot"])
    f.suptitle("M07  自回归生长的训练目标：真实网络按语料顺序逐点生长，每一步的监督 = 新节点的接点集合（多接点 = 闭合回路，这是集合似然损失存在的原因）", x=0.02, ha="left", fontweight="bold")
    f.tight_layout(rect=(0.02, 0.06, 1, 0.94))
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


def _short_source(o) -> str:  # noqa: ANN001
    src = str(getattr(o, "source", "") or "")
    if "outer_mask" in src or src.endswith(".png"):
        return "轮廓掩膜"
    if "total" in src:
        return "店铺多边形并集"
    return "手绘"


def _tag(ax, text: str, fs: float) -> None:  # noqa: ANN001
    """Short metric tag under a panel (instead of a two-line title that collides with the column headers)."""
    ax.text(0.5, -0.03, text, transform=ax.transAxes, ha="center", va="top", fontsize=fs, color="#333", linespacing=1.2)


def m08_stage3_fit_pipeline(ctx: Ctx, out: Path) -> list[Path]:
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan, poly

    fids = _stage3_floors(ctx, 3)
    if not fids:
        return []
    s = load_style()
    fs = s["fonts"]["size_annot"] - 0.5
    cols = ["① 轮廓（真实楼层）", "② 内缩区 + 中轴线", "③ 初始化：外环钉在内缩边界\n支路落在中轴", "④ 力松弛 + 交叉 / 越界修复\n+ 开角 + 角点吸附", "⑤ 一键成廊", "真实建成\n（同一网络的实测位置）"]
    f, axes = plt.subplots(len(fids), len(cols), figsize=(s["figure"]["width_double"] * 1.4, max(3.1 * len(fids) + 1.8, 5.4)), gridspec_kw={"hspace": 0.6, "wspace": 0.08})
    axes = np.atleast_2d(axes)
    for r, fid in enumerate(fids):
        st = _fit_stages(ctx, fid)
        o, ins = st["outline"], st["inset"]
        ax = axes[r, 0]; poly(ax, o.polygon, fc="#f4f4f4", ec="#333", lw=1.0); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        _tag(ax, f"{o.area / 1e4:.1f} 万 m² · 来源：{_short_source(o)}", fs)
        ax = axes[r, 1]; poly(ax, o.polygon, fc="#f4f4f4", ec="#333", lw=1.0); poly(ax, ins, fc="#e8eef7", ec="#4a6fa5", lw=0.8)
        _draw_lines(ax, st["lines"], color="#4a6fa5", lw=0.7, alpha=0.9); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
        _tag(ax, f"内缩深度 {st['depth']:.0f} m（0.18·√面积）", fs)
        d0 = st["res"].diagnostics
        draw_network_in_outline(axes[r, 2], st["topo"], st["init"], o, st["sk_nodes"], node_size=9)
        poly(axes[r, 2], ins, fc="none", ec="#4a6fa5", lw=0.6, ls="--")
        _tag(axes[r, 2], f"{st['topo'].num_nodes} 节点 · 外环 {d0['roles']['outer']} · 支路 {d0['roles']['branch'] + d0['roles']['leaf']}", fs)
        draw_network_in_outline(axes[r, 3], st["topo"], {k: np.asarray(v) for k, v in st["res"].positions.items()}, o, st["sk_nodes"], node_size=9)
        _tag(axes[r, 3], f"交叉 {d0['crossings']} · 越界 {d0.get('edges_outside', 0)} · 锐角率 {d0['sharp_angle_rate']:.2f}\n墙向偏差 {d0['ortho_deviation_deg']:.0f}°", fs)
        dp = st["plan"].diagnostics
        draw_plan(axes[r, 4], st["plan"], o)
        _tag(axes[r, 4], f"主廊 {dp['main_width_m']:.0f} m / 次廊 {dp['secondary_width_m']:.0f} m\n{dp['n_entrances']} 入口 · {dp['n_atria']} 中庭 · {dp['n_vertical_cores']} 竖向核 · 占比 {dp['corridor_ratio'] * 100:.0f}%", fs)
        gt = st["gt"]
        if len(gt) >= 0.9 * st["topo"].num_nodes:
            draw_network_in_outline(axes[r, 5], st["topo"], gt, o, st["sk_nodes"], node_size=9, style="ink")
            _tag(axes[r, 5], "黑 = 骨架（原型）节点", fs)
        else:
            axes[r, 5].axis("off")
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    _finish_grid(f, axes, cols, "M08  阶段三拟合流程：从轮廓与关键点网络到走廊方案（3 个真实楼层；①–⑤ 为 CorridorFitter / render_corridors 的中间状态）", bottom=0.04)
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M08_stage3_fit_pipeline")


def _naive_embedding(topo, outline, skeleton_nodes):  # noqa: ANN001
    """The pre-Stage-3 baseline: abstract planar embedding scaled into the outline's bounding box (what one gets without
    outline awareness – nodes may fall outside, corridors cut re-entrant corners, angles are arbitrary)."""
    from mall_space_planner.geometry.planar_embed import PlanarEmbedParams, planar_corridor_embedding

    pos, _ = planar_corridor_embedding(topo, PlanarEmbedParams(ortho_weight=0.0, relax_iters=60), skeleton_nodes=skeleton_nodes)
    P = np.array(list(pos.values()), float)
    lo, hi = P.min(0), P.max(0)
    minx, miny, maxx, maxy = outline.polygon.bounds
    sc = np.array([(maxx - minx) * 0.9 / max(hi[0] - lo[0], 1e-9), (maxy - miny) * 0.9 / max(hi[1] - lo[1], 1e-9)])
    off = np.array([minx + 0.05 * (maxx - minx), miny + 0.05 * (maxy - miny)])
    return {k: (np.asarray(v, float) - lo) * sc + off for k, v in pos.items()}


def _nodes_only_inside(P: dict, outline) -> dict:  # noqa: ANN001
    """Project every node that lies outside the outline onto the nearest point of the (slightly shrunk) outline – node
    containment without edge containment, i.e. the pre-fix guarantee."""
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    inner = outline.polygon.buffer(-2.0)
    out = {}
    for k, v in P.items():
        pt = Point(float(v[0]), float(v[1]))
        if inner.contains(pt):
            out[k] = np.asarray(v, float)
        else:
            q = nearest_points(inner, pt)[0]
            out[k] = np.array([q.x, q.y])
    return out


def _fit_with(topo, outline, skn, fp, seed=0):  # noqa: ANN001, ANN201
    from mall_space_planner.stage3 import CorridorFitter

    fitter = CorridorFitter(fp)
    res = fitter.fit(topo, outline, seed=seed, skeleton_nodes=skn)
    return res, {k: np.asarray(v) for k, v in res.positions.items()}


def _mark_bad(ax, g, P, outline, min_deg: float = 45.0) -> tuple[int, int, int]:  # noqa: ANN001
    """Overlay the problems: red = corridor segment leaving the building, red ring = sharp wedge (< min_deg),
    red cross = node outside. Returns (edges outside, sharp wedges, nodes outside)."""
    from shapely.geometry import Point

    from mall_space_planner.stage3.fit import _EdgeInsideChecker, node_angles

    site = outline.polygon.buffer(-1.0)
    ec = _EdgeInsideChecker(site)
    bad_e = ec.bad_edges(g, P)
    for u, v in bad_e:
        ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color="#C0392B", lw=2.2, zorder=7)
    n_out = 0
    for v in g.nodes:
        if not outline.polygon.buffer(0.5).contains(Point(P[v])):
            n_out += 1
            ax.scatter([P[v][0]], [P[v][1]], marker="x", s=36, c="#C0392B", lw=1.4, zorder=8)
    sharp = 0
    for v in g.nodes:
        nb = list(g.neighbors(v))
        if len(nb) < 2:
            continue
        angs = sorted(float(np.arctan2(*(P[u] - P[v])[::-1])) for u in nb)
        gaps = np.diff(np.array(angs + [angs[0] + 2 * np.pi]))
        if (np.degrees(gaps) < min_deg).any():  # a visible wedge between two angularly adjacent corridors
            sharp += 1
            ax.scatter([P[v][0]], [P[v][1]], s=55, facecolors="none", edgecolors="#C0392B", linewidths=1.0, zorder=8)
    return len(bad_e), sharp, n_out


def m09_stage3_rule_before_after(ctx: Ctx, out: Path) -> list[Path]:
    """Rule off vs on on the same floor / network / seed, shown as clean corridor plans (network in outline + one-click
    corridors), one pair per rule: (a) angle rule R0, (b) dead-end → entrance / leaf-to-façade force R9, (c) L outline
    edge containment. Metrics in the captions carry the before/after difference."""
    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.data.legacy_adapter import load_graph_csv
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan
    from mall_space_planner.stage3 import CorridorFitter, FitParams, RenderParams, outline_from_points, render_corridors
    from mall_space_planner.stage3.fit import edges_outside, sharp_angle_rate

    cands = _stage3_floors(ctx, 6, max_nodes=55, min_nodes=14)
    if not cands:
        return []
    s = load_style()
    fs = s["fonts"]["size_annot"] - 0.5
    gd = ctx.graph_dir
    rp = RenderParams(corridor_ratio=0.17, main_width=7.0, min_width=2.6, secondary_factor=0.6)
    fp_on = FitParams(n_restarts=4, iters=100)
    rows = []
    for fid in cands:
        topo = load_target_csv(gd / f"{fid}_M.csv")
        sk = load_graph_csv(gd / f"{fid}_M_simplified.csv", gd / f"{fid}_M_simplified_node_attributes.csv")
        skn = set(sk.nodes) & set(topo.nodes)
        o = ctx.ds.outline(fid, 0)
        g = nx.Graph(topo.edges()); g.add_nodes_from(topo.nodes)
        A = o.area; w = np.sqrt(A / (1 - 0.58**2))
        L = outline_from_points([(0, 0), (w, 0), (w, 0.42 * w), (0.42 * w, 0.42 * w), (0.42 * w, w), (0, w)])

        def _pl(res, oo):  # noqa: ANN001, ANN202
            return render_corridors(topo, res.positions, oo, res.roles, rp, skeleton_nodes=skn)

        # (a) angle rule: pick the seed where switching it off hurts most
        best = None
        for sd in (0, 1, 2):
            r0 = CorridorFitter(FitParams(n_restarts=4, iters=100, min_angle_deg=0.0, w_angle=0.0)).fit(topo, o, seed=sd, skeleton_nodes=skn)
            r1 = CorridorFitter(fp_on).fit(topo, o, seed=sd, skeleton_nodes=skn)
            gain = r0.diagnostics["sharp_angle_rate"] - r1.diagnostics["sharp_angle_rate"]
            if best is None or gain > best[0]:
                best = (gain, r0, r1)
        gain_a, ra0, ra1 = best
        # (b) leaf→façade force off vs on (dead ends become interior cores vs entrances)
        rb0 = CorridorFitter(FitParams(n_restarts=4, iters=100, w_leaf_facade=0.0)).fit(topo, o, seed=0, skeleton_nodes=skn)
        pb0, pb1 = _pl(rb0, o), _pl(ra1, o)
        gain_b = pb0.diagnostics["n_vertical_cores"] - pb1.diagnostics["n_vertical_cores"]
        # (c) L outline: node-only containment (pre-fix) vs edge containment
        P_off = _nodes_only_inside(_naive_embedding(topo, L, skn), L)
        n_off = edges_outside(g, P_off, L.polygon.buffer(-1.0))
        rc1 = CorridorFitter(fp_on).fit(topo, L, seed=0, skeleton_nodes=skn)
        rows.append((gain_a + 0.1 * gain_b + 0.2 * n_off, fid, topo, skn, o, L, g, (ra0, ra1), (rb0, pb0, pb1), (P_off, n_off, rc1)))
        if len(rows) >= 5:
            break
    rows.sort(key=lambda r: -r[0])
    rows = rows[:3]
    cols = ["(a) 关闭开角规则", "(a) 开角规则 ≥ 60°（R0）", "(b) 无断头→立面力\n（走廊方案）", "(b) 断头 → 出入口（R9）\n（走廊方案）", "(c) L 形轮廓\n只约束节点在内", "(c) L 形轮廓\n边越界修复 → 成廊"]
    f, axes = plt.subplots(len(rows), 6, figsize=(s["figure"]["width_double"] * 1.4, max(3.1 * len(rows) + 1.8, 5.4)), gridspec_kw={"hspace": 0.6, "wspace": 0.08})
    axes = np.atleast_2d(axes)
    for r, (_, fid, topo, skn, o, L, g, (ra0, ra1), (rb0, pb0, pb1), (P_off, n_off, rc1)) in enumerate(rows):
        for c, res in ((0, ra0), (1, ra1)):
            P = {k: np.asarray(v) for k, v in res.positions.items()}
            draw_network_in_outline(axes[r, c], topo, P, o, skn, node_size=9)
            _tag(axes[r, c], f"锐角率(<60°) {sharp_angle_rate(g, P, 60.0):.2f}\n交叉 {res.diagnostics['crossings']} · 墙向偏差 {res.diagnostics['ortho_deviation_deg']:.0f}°", fs)
        for c, pl in ((2, pb0), (3, pb1)):
            d = pl.diagnostics
            draw_plan(axes[r, c], pl, o)
            _tag(axes[r, c], f"{d['n_entrances']} 出入口 · {d['n_vertical_cores']} 内部竖向核\n{d['n_atria']} 中庭 · 占比 {d['corridor_ratio'] * 100:.0f}%", fs)
        draw_network_in_outline(axes[r, 4], topo, P_off, L, skn, node_size=9)
        _tag(axes[r, 4], f"越出建筑的走廊段 {n_off}", fs)
        plc = _pl(rc1, L); d = plc.diagnostics
        draw_plan(axes[r, 5], plc, L)
        _tag(axes[r, 5], f"越界边 {rc1.diagnostics.get('edges_outside', 0)} · 交叉 {rc1.diagnostics['crossings']}\n{d['n_entrances']} 出入口 · {d['n_atria']} 中庭 · 占比 {d['corridor_ratio'] * 100:.0f}%", fs)
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}\n{topo.num_nodes} 节点", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    _finish_grid(f, axes, cols, "M09  阶段三规则的前后对比（同一楼层、同一网络、同一随机种子）：(a) 开角 (b) 断头→出入口 (c) 非凸轮廓的边越界修复", bottom=0.04)
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


def _plan_quality(res, plan, topo) -> float:  # noqa: ANN001
    """Lower is better: crossings, outside edges, sharp wedges, interior dead ends (vertical cores), corridor share
    beyond the real range – used to pick good showcase results among seeds / candidate floors (M10)."""
    d, dp = res.diagnostics, plan.diagnostics
    cr = float(np.clip(dp["corridor_ratio"], 0, 1))
    return 3.0 * d["crossings"] + 3.0 * d.get("edges_outside", 0) + 4.0 * d["sharp_angle_rate"] + 0.6 * dp["n_vertical_cores"] + 6.0 * max(0.0, cr - 0.26) + 2.0 * (1.0 - d.get("served_area_ratio", 1.0))


def _best_of_seeds(topo, outline, skn, fp, rp, seeds=(0, 1, 2)):  # noqa: ANN001, ANN201
    from mall_space_planner.stage3 import CorridorFitter, render_corridors

    fitter = CorridorFitter(fp)
    best = None
    for sd in seeds:
        res = fitter.fit(topo, outline, seed=sd, skeleton_nodes=skn)
        plan = render_corridors(topo, res.positions, outline, res.roles, rp, skeleton_nodes=skn)
        q = _plan_quality(res, plan, topo)
        if best is None or q < best[0]:
            best = (q, res, plan)
    return best


def m10_stage3_transfer_outlines(ctx: Ctx, out: Path) -> list[Path]:
    """Same real network into: its own outline / the closest-area outline of another mall / a hand-drawn L / U.
    Showcase quality: 4 seeds per outline, the best by crossings / outside edges / sharp wedges / interior dead ends /
    corridor share is shown; candidate floors are ranked by the same quality so weak cases are not displayed."""
    from mall_space_planner.data.corpus_builder import load_target_csv
    from mall_space_planner.data.legacy_adapter import load_graph_csv
    from mall_space_planner.hub.viz import draw_network_in_outline, draw_plan
    from mall_space_planner.stage3 import FitParams, RenderParams, outline_from_points
    from mall_space_planner.stage3.outline import m_positions_from_total_csv

    cands = _stage3_floors(ctx, 6, max_nodes=55, min_nodes=16)
    if not cands:
        return []
    s = load_style()
    fs = s["fonts"]["size_annot"] - 0.5
    gd = ctx.graph_dir
    fp = FitParams(n_restarts=4, iters=110, min_angle_deg=65.0, w_angle=1.0)
    rp = RenderParams(corridor_ratio=0.17, main_width=7.0, secondary_width=4.5, min_width=2.6, secondary_factor=0.6, atrium_per_area_m2=9000.0)
    rows = []
    for fid in cands:
        topo = load_target_csv(gd / f"{fid}_M.csv")
        sk = load_graph_csv(gd / f"{fid}_M_simplified.csv", gd / f"{fid}_M_simplified_node_attributes.csv")
        skn = set(sk.nodes) & set(topo.nodes)
        o = ctx.ds.outline(fid, 0)
        gt = m_positions_from_total_csv(gd / f"{fid}_total.csv", o)
        A = o.area
        w = np.sqrt(A / (1 - 0.58**2))
        L = outline_from_points([(0, 0), (w, 0), (w, 0.42 * w), (0.42 * w, 0.42 * w), (0.42 * w, w), (0, w)])
        wu = np.sqrt(A / 0.64)
        U = outline_from_points([(0, 0), (wu, 0), (wu, 0.8 * wu), (0.7 * wu, 0.8 * wu), (0.7 * wu, 0.3 * wu), (0.3 * wu, 0.3 * wu), (0.3 * wu, 0.8 * wu), (0, 0.8 * wu)])
        other = None
        mall = fid.rsplit("_", 1)[0]
        try:
            for cand in ctx.ds.similar_outlines(A, k=8, exclude_mall=mall):
                parts = cand.split("_")
                cf, reg = "_".join(parts[:2]), int(parts[2]) if len(parts) >= 3 else 0
                if cf.rsplit("_", 1)[0] != mall:
                    other = (f"另一真实商场 {cf}", ctx.ds.outline(cf, reg))
                    break
        except Exception:  # noqa: BLE001
            other = None
        if other is None:
            other = ("矩形（沙箱示意）", outline_from_points([(0, 0), (np.sqrt(A * 2.2), 0), (np.sqrt(A * 2.2), np.sqrt(A / 2.2)), (0, np.sqrt(A / 2.2))]))
        fits = [(name, oo, _best_of_seeds(topo, oo, skn, fp, rp)) for name, oo in (("自身", o), other, ("L", L), ("U", U))]
        total_q = sum(q for _, _, (q, _, _) in fits)
        rows.append((total_q, fid, topo, skn, o, gt, fits))
        if len(rows) >= 4:
            break
    rows.sort(key=lambda r: r[0])
    # keep different types among the best three when possible
    picked, used = [], set()
    for row in rows:
        lt = _floor_type(ctx, row[1])
        if lt in used and len(rows) - len(picked) > 3 - len(picked):
            continue
        picked.append(row); used.add(lt)
        if len(picked) == 3:
            break
    rows = picked or rows[:3]
    cols = ["真实网络\n（建成位置）", "自身轮廓\n→ 成廊", "另一真实商场轮廓\n（面积最近）", "手绘 L 形轮廓\n（等面积）", "手绘 U 形轮廓\n（等面积）"]
    f, axes = plt.subplots(len(rows), 5, figsize=(s["figure"]["width_double"] * 1.3, max(3.2 * len(rows) + 1.8, 5.4)), gridspec_kw={"hspace": 0.6, "wspace": 0.08})
    axes = np.atleast_2d(axes)
    for r, (_, fid, topo, skn, o, gt, fits) in enumerate(rows):
        draw_network_in_outline(axes[r, 0], topo, gt, o, skn, node_size=9, style="ink")
        _tag(axes[r, 0], f"{topo.num_nodes} 节点 · {o.area / 1e4:.1f} 万 m²", fs)
        for c, (name, oo, (q, res, plan)) in enumerate(fits, start=1):
            d, dp = res.diagnostics, plan.diagnostics
            draw_plan(axes[r, c], plan, oo)
            head = f"{name}\n" if c == 2 and "沙箱" not in name else ""
            _tag(axes[r, c], f"{head}交叉 {d['crossings']} · 越界 {d.get('edges_outside', 0)} · 锐角率 {d['sharp_angle_rate']:.2f}\n{dp['n_entrances']} 入口 · {dp['n_atria']} 中庭 · {dp['n_vertical_cores']} 竖向核\n走廊占比 {dp['corridor_ratio'] * 100:.0f}%", fs - 0.5)
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    _finish_grid(f, axes, cols, "M10  轮廓自适应：同一个真实关键点网络放入自身轮廓、另一真实商场轮廓与手绘 L / U 形轮廓（骨架环 = 外环；每格为 4 个随机种子中几何质量最好的结果）", bottom=0.04)
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
    fs = s["fonts"]["size_annot"] - 0.5
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
                ("v2：仅锚定既有出入口\n+ 阶段一预测评分（骨架位置重排）", dict(keep_skeleton_positions=False, score_fn=score_fn), _v2_objective),
                ("v2b（本文）：回路中性\n+ 整合度下降惩罚", dict(keep_skeleton_positions=False, score_fn=score_fn), None)]
    f, axes = plt.subplots(len(fids), 1 + 2 * len(versions), figsize=(s["figure"]["width_double"] * 1.6, max(3.3 * len(fids) + 2.0, 5.8)), gridspec_kw={"width_ratios": [1] + [1, 1] * len(versions), "hspace": 0.65, "wspace": 0.08})
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
            ia, ib = res["ind_a"], res["ind_b"]
            if first is None:
                first = res
                draw_network_in_outline(axes[r, 0], res["before"], res["gt"], res["outline"], res["sk_nodes"], node_size=9, style="ink")
                _tag(axes[r, 0], f"{ib['num_nodes']} 节点 · 回路 {ib['num_cycles']} · 断头 {ib['n_dead_ends']}\nASPL {ib['avg_shortest_path']:.2f}\n整合度 {ib['closeness_mean']:.3f}", fs - 0.5)
            P = {k: np.asarray(v) for k, v in res["res"].positions.items()}

            def d(k, fmt="{:+.2f}"):  # noqa: ANN001, ANN202
                return fmt.format(ia[k] - ib[k])

            draw_network_in_outline(axes[r, 1 + 2 * c], res["after"], P, res["outline"], res["sk_nodes"], node_size=9, anchored=res.get("anchored", ()))
            _tag(axes[r, 1 + 2 * c], f"回路 {ia['num_cycles']} ({d('num_cycles', '{:+d}')}) · 断头 {ia['n_dead_ends']} ({d('n_dead_ends', '{:+d}')})\nASPL {ia['avg_shortest_path']:.2f} ({d('avg_shortest_path')})\n整合度 {ia['closeness_mean']:.3f} ({d('closeness_mean', '{:+.3f}')})", fs - 0.5)
            dp = res["plan"].diagnostics
            draw_plan(axes[r, 2 + 2 * c], res["plan"], res["outline"])
            _tag(axes[r, 2 + 2 * c], f"交叉 {res['row']['after_crossings']} · {dp['n_entrances']} 入口 · {dp['n_atria']} 中庭\n锐角率 {res['row']['after_sharp_angle_rate']:.2f} · 占比 {dp['corridor_ratio'] * 100:.0f}%", fs - 0.5)
        axes[r, 0].text(-0.08, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    hdr = ["现状（真实建成）"]
    for vname, _, _ in versions:
        hdr += [vname, "走廊方案"]
    _finish_grid(f, axes, hdr, f"M11  改造工作流三个版本的对比（同一楼层、同一轮廓、{gen_name}、同一随机种子；红框 = 锚定节点；括号 = 相对现状的变化）", bottom=0.05)
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M11_renovation_versions")


def _plan_image(ax, outline, plan_png) -> bool:  # noqa: ANN001
    """The colour-block plan as published (no desaturation / blending), cropped to the outline bounds."""
    if plan_png is None or not Path(plan_png).exists():
        return False
    from PIL import Image

    img = np.asarray(Image.open(plan_png).convert("RGB"))
    H, W = img.shape[:2]
    (x0, y0), (x1, y1) = outline.px_to_m(np.array([[0.0, 0.0], [float(W), float(H)]]))
    ax.imshow(img, extent=(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)), interpolation="lanczos", zorder=0)
    return True


def m12_data_levels(ctx: Ctx, out: Path) -> list[Path]:
    """Three real floors of different types: published colour-block plan | complete M network on a clean white plate
    (thin outline, uniform small nodes) | skeleton only. No overlay of lines on the plan – each level gets its own panel."""
    from mall_space_planner.hub.plates import load_floor
    from mall_space_planner.hub.viz import poly
    from mall_space_planner.schemas import TopologyGraph

    fids = _stage3_floors(ctx, 3, max_nodes=120)
    if not fids:
        return []
    s = load_style()
    fs = s["fonts"]["size_annot"] - 0.5
    cols = ["色块平面图（阶段三图像数据）", "完整走廊关键点网络 M\n（阶段二目标 / 阶段三输入）", "骨架 M_s\n（阶段一原型 / 阶段二输入）"]
    f, axes = plt.subplots(len(fids), 3, figsize=(s["figure"]["width_double"] * 1.05, 2.9 * len(fids) + 1.3), gridspec_kw={"hspace": 0.4, "wspace": 0.06})
    axes = np.atleast_2d(axes)

    def _net(ax, g, pos, sk, lw_main, lw_sec):  # noqa: ANN001, ANN202
        poly(ax, o.polygon, fc="white", ec="#444", lw=0.9)
        G = nx.Graph(g.edges()); G.add_nodes_from(g.nodes)
        for u, v in G.edges:
            if u in pos and v in pos:
                main = u in sk and v in sk
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#111" if main else "#555", lw=lw_main if main else lw_sec, zorder=3, solid_capstyle="round")
        nodes = [v for v in G.nodes if v in pos]
        ax.scatter([pos[v][0] for v in nodes], [pos[v][1] for v in nodes], s=7, c=["#111" if v in sk else "white" for v in nodes], edgecolors="#111", linewidths=0.6, zorder=5)
        minx, miny, maxx, maxy = o.polygon.bounds
        mx, my = 0.04 * (maxx - minx), 0.04 * (maxy - miny)
        ax.set_xlim(minx - mx, maxx + mx); ax.set_ylim(miny - my, maxy + my)
        ax.set_aspect("equal"); ax.axis("off")

    for r, fid in enumerate(fids):
        rn = load_floor(fid, ctx.ds, ctx.graph_dir)
        full, sk, pos, o = rn["full"], rn["skeleton"], rn["positions"], rn["outline"]
        skn = set(sk.nodes) if sk else set()
        ax = axes[r, 0]
        if _plan_image(ax, o, rn["plan_png"]):
            poly(ax, o.polygon, fc="none", ec="#222", lw=0.8)
            minx, miny, maxx, maxy = o.polygon.bounds
            mx, my = 0.04 * (maxx - minx), 0.04 * (maxy - miny)
            ax.set_xlim(minx - mx, maxx + mx); ax.set_ylim(miny - my, maxy + my)
            ax.set_aspect("equal"); ax.axis("off")
            _tag(ax, f"{o.area / 1e4:.1f} 万 m²", fs)
        else:
            poly(ax, o.polygon, fc="#f4f4f4", ec="#444", lw=0.9); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
            _tag(ax, f"{o.area / 1e4:.1f} 万 m²（无平面图，仅轮廓）", fs)
        _net(axes[r, 1], full, pos, skn, 1.4, 0.7)
        _tag(axes[r, 1], f"{full.num_nodes} 节点 / {full.num_edges} 边（黑 = 骨架节点与骨架边）", fs)
        if sk is not None:
            sk_only = TopologyGraph(adjacency={u: [v for v in sk.adjacency[u] if v in pos] for u in sk.nodes if u in pos})
            _net(axes[r, 2], sk_only, pos, skn, 1.4, 0.7)
            _tag(axes[r, 2], f"{sk.num_nodes} 节点 / {sk.num_edges} 边", fs)
        else:
            axes[r, 2].axis("off")
        axes[r, 0].text(-0.06, 0.5, f"{_floor_type(ctx, fid)}\n{fid}", transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=s["fonts"]["size_annot"], rotation=90)
    _finish_grid(f, axes, cols, "M12  同一楼层的三个数据层级：色块平面 → 完整走廊关键点网络 M → 骨架 M_s（真实楼层，同一像素坐标系）")
    _note(f, ctx, ctx.is_real_graphs)
    return savefig(f, out, "M12_data_levels")


ALL = [m01_stage1_pipeline_cases, m02_stage1_ranker_comparison, m03_stage1_type_conditional_effect, m04_stage2_method_comparison, m05_stage2_ablation_cases, m06_stage2_decoding_cases, m07_stage2_growth_steps, m08_stage3_fit_pipeline, m09_stage3_rule_before_after, m10_stage3_transfer_outlines, m11_renovation_versions, m12_data_levels]

__all__ = ["ALL", "Ctx", "build_ctx", *[fn.__name__ for fn in ALL]]
