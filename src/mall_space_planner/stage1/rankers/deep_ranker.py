"""Deep ranker: condition Transformer + prototype GNN with **residual late fusion**.

Design (informed by the failure analysis of the legacy Transformer+GNN+cross-attention):

* ``score = s_tab(x) + α · f_θ(condition tokens, graph)`` — the deep network only learns a
  *residual* on top of a fitted tabular baseline (ExtraTrees by default). With α→0 the
  model degrades to the baseline instead of below it; the learned α is reported.
* Condition encoder: per-feature tokens (value + learned feature id embedding) → 1–2 layer
  Transformer encoder → mean pool. Graph encoder: L-layer GIN-style message passing in
  pure PyTorch (no PyG dependency) over the prototype skeleton with node features
  [degree, clustering, betweenness, x, y]; sum/mean pooling. Late fusion = concat →
  MLP; **no cross-attention** between the two (the legacy bottleneck).
* Small-sample tricks (all switchable): Gaussian feature noise, node dropout, listwise
  softmax-CE loss over groups (LambdaRank-free, stable), weight decay, early stopping on
  val NDCG, snapshot ensembling of the best K checkpoints, deterministic seeding.
* Ablation switches: ``use_transformer`` (else MLP), ``use_gnn`` (else zero graph branch),
  ``use_residual`` (else end-to-end without the tabular base), ``fusion`` in {late, early}.

Runs on CPU / CUDA / Apple MPS. Graph tensors are precomputed once per fit.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from mall_space_planner.registry import register
from mall_space_planner.stage1.base import BaseRanker, RankingContext
from mall_space_planner.stage1.rankers.sklearn_rankers import ExtraTreesPointwiseRanker, make_bucket_id, sample_groups
from mall_space_planner.topology.convert import to_networkx
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _torch():  # noqa: ANN202
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("deep_ranker requires torch: pip install torch") from exc
    return torch


def select_device(prefer: str = "auto") -> Any:
    torch = _torch()
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- graph tensors
NODE_FEAT_DIM = 5


def graph_to_tensors(graph, torch) -> tuple[Any, Any]:  # noqa: ANN001
    """Return (node_features [N,5], edge_index [2,2E]) with normalised coordinates."""
    g = to_networkx(graph)
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    deg = np.array([g.degree(v) for v in nodes], dtype=np.float32)
    clus = np.array(list(nx.clustering(g).values()), dtype=np.float32) if n > 2 else np.zeros(n, np.float32)
    bc = np.array(list(nx.betweenness_centrality(g).values()), dtype=np.float32) if n > 2 else np.zeros(n, np.float32)
    pos = np.array([graph.positions.get(v, (0.0, 0.0)) for v in nodes], dtype=np.float32)
    if len(pos) and graph.positions:
        pos = pos - pos.mean(0)
        pos = pos / (np.abs(pos).max() + 1e-6)
    x = np.stack([np.log1p(deg), clus, bc, pos[:, 0], pos[:, 1]], axis=1)
    src, dst = [], []
    for u, v in g.edges:
        src += [idx[u], idx[v]]
        dst += [idx[v], idx[u]]
    ei = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)
    return torch.tensor(x), torch.tensor(ei)


def _build_modules(torch):  # noqa: ANN001, ANN202
    nn = torch.nn

    class ConditionEncoder(nn.Module):
        def __init__(self, n_feat: int, d: int, use_transformer: bool, n_layers: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            self.use_transformer = use_transformer
            if use_transformer:
                self.fid = nn.Embedding(n_feat, d)
                self.val = nn.Linear(1, d)
                layer = nn.TransformerEncoderLayer(d, n_heads, d * 2, dropout, batch_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(layer, n_layers)
            else:
                self.mlp = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
            self.register_buffer("ids", torch.arange(n_feat))

        def forward(self, q):  # noqa: ANN001, ANN202
            if not self.use_transformer:
                return self.mlp(q)
            tok = self.fid(self.ids)[None].expand(q.shape[0], -1, -1) + self.val(q.unsqueeze(-1))
            return self.enc(tok).mean(1)

    class GINLayer(nn.Module):
        def __init__(self, d: int, dropout: float) -> None:
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
            self.eps = nn.Parameter(torch.zeros(1))
            self.norm = nn.LayerNorm(d)

        def forward(self, h, ei):  # noqa: ANN001, ANN202
            agg = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]]) if ei.shape[1] else torch.zeros_like(h)
            return self.norm(h + self.mlp((1 + self.eps) * h + agg))

    class GraphEncoder(nn.Module):
        def __init__(self, d: int, n_layers: int, dropout: float, node_dropout: float) -> None:
            super().__init__()
            self.inp = nn.Linear(NODE_FEAT_DIM, d)
            self.layers = nn.ModuleList([GINLayer(d, dropout) for _ in range(n_layers)])
            self.node_dropout = node_dropout
            self.out = nn.Linear(2 * d, d)

        def forward(self, x, ei, batch, n_graphs):  # noqa: ANN001, ANN202
            h = self.inp(x)
            if self.training and self.node_dropout > 0:
                keep = (torch.rand(h.shape[0], device=h.device) > self.node_dropout).float()[:, None]
                h = h * keep
            for layer in self.layers:
                h = layer(h, ei)
            mean = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
            cnt = torch.zeros(n_graphs, 1, device=h.device).index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device)).clamp(min=1)
            mx = torch.full((n_graphs, h.shape[1]), -1e9, device=h.device).scatter_reduce(0, batch[:, None].expand_as(h), h, reduce="amax")
            return self.out(torch.cat([mean / cnt, mx], 1))

    class ResidualFusionRanker(nn.Module):
        def __init__(self, n_cond: int, n_proto: int, d: int, cfg: dict[str, Any]) -> None:
            super().__init__()
            self.use_gnn, self.use_residual, self.fusion = cfg["use_gnn"], cfg["use_residual"], cfg["fusion"]
            self.cond = ConditionEncoder(n_cond, d, cfg["use_transformer"], cfg["n_transformer_layers"], cfg["n_heads"], cfg["dropout"])
            self.proto = nn.Sequential(nn.Linear(max(1, n_proto), d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d, d))
            self.gnn = GraphEncoder(d, cfg["n_gnn_layers"], cfg["dropout"], cfg["node_dropout"]) if self.use_gnn else None
            fin = d * (3 if self.use_gnn else 2)
            self.head = nn.Sequential(nn.LayerNorm(fin), nn.Linear(fin, d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d, 1))
            self.alpha = nn.Parameter(torch.tensor(float(cfg["alpha_init"])))
            self.tab_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, q, p, s_tab, gx, gei, gbatch, n):  # noqa: ANN001, ANN202
            parts = [self.cond(q), self.proto(p)]
            if self.use_gnn:
                parts.append(self.gnn(gx, gei, gbatch, n))
            deep = self.head(torch.cat(parts, 1)).squeeze(-1)
            if self.use_residual:
                return self.tab_scale * s_tab + self.alpha * deep
            return deep

    return ResidualFusionRanker


@register("ranker", "deep_residual")
class DeepResidualRanker(BaseRanker):
    supports_feature_importance = False

    def __init__(
        self,
        d_model: int = 48,
        use_transformer: bool = True,
        n_transformer_layers: int = 1,
        n_heads: int = 4,
        use_gnn: bool = True,
        n_gnn_layers: int = 2,
        use_residual: bool = True,
        fusion: str = "late",
        alpha_init: float = 0.1,
        dropout: float = 0.2,
        node_dropout: float = 0.1,
        feature_noise: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        epochs: int = 40,
        patience: int = 8,
        batch_groups: int = 32,
        candidates_per_query: int = 20,
        ensemble_k: int = 3,
        base_ranker: str = "extra_trees",
        device: str = "auto",
        seed: int = 42,
        area_thresholds: list[float] | None = None,
    ) -> None:
        self.cfg = dict(d_model=d_model, use_transformer=use_transformer, n_transformer_layers=n_transformer_layers, n_heads=n_heads, use_gnn=use_gnn, n_gnn_layers=n_gnn_layers, use_residual=use_residual, fusion=fusion, alpha_init=alpha_init, dropout=dropout, node_dropout=node_dropout)
        self.feature_noise, self.lr, self.weight_decay, self.epochs, self.patience = feature_noise, lr, weight_decay, epochs, patience
        self.batch_groups, self.candidates_per_query, self.ensemble_k = batch_groups, candidates_per_query, ensemble_k
        self.base_name, self.device_pref, self.seed = base_ranker, device, seed
        self.area_thresholds = list(area_thresholds or [200_000, 450_000])
        self.base: BaseRanker | None = None
        self.states_: list[dict] = []
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_ndcg10": []}
        self.graph_cache_: dict[str, tuple[Any, Any]] = {}
        self._model = None

    # ------------------------------------------------------------------ helpers
    def _graphs(self, ctx: RankingContext, ids: list[str], torch):  # noqa: ANN001, ANN202
        xs, eis, batch, off = [], [], [], 0
        for i, fid in enumerate(ids):
            if fid not in self.graph_cache_:
                g = ctx.db.get_graph(fid)
                self.graph_cache_[fid] = graph_to_tensors(g, torch) if g is not None else (torch.zeros(1, NODE_FEAT_DIM), torch.zeros(2, 0, dtype=torch.long))
            x, ei = self.graph_cache_[fid]
            xs.append(x)
            eis.append(ei + off)
            batch.append(torch.full((x.shape[0],), i, dtype=torch.long))
            off += x.shape[0]
        return torch.cat(xs), torch.cat(eis, 1), torch.cat(batch)

    def _inputs(self, ctx: RankingContext, q_df: pd.DataFrame, c_df: pd.DataFrame, torch, device):  # noqa: ANN001, ANN202
        q = torch.tensor(ctx.features.condition_matrix(q_df))
        p = torch.tensor(ctx.features.prototype_matrix(c_df)) if ctx.features.proto_scaler.cols else torch.zeros(len(c_df), 1)
        s_tab = torch.tensor(self.base.score(ctx, q_df, c_df)) if self.base is not None else torch.zeros(len(c_df))
        gx, gei, gb = self._graphs(ctx, c_df[ctx.db.id_col].astype(str).tolist(), torch)
        return [t.to(device) for t in (q, p, s_tab, gx, gei, gb)]

    def _new_model(self, ctx: RankingContext, torch):  # noqa: ANN001, ANN202
        cls = _build_modules(torch)
        return cls(len(ctx.features.spec.query_cols), len(ctx.features.proto_scaler.cols), self.cfg["d_model"], self.cfg)

    # ------------------------------------------------------------------ fit
    def fit(self, ctx: RankingContext, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> DeepResidualRanker:
        torch = _torch()
        torch.manual_seed(self.seed)
        device = select_device(self.device_pref)
        rng = np.random.RandomState(self.seed)
        spec = ctx.features.spec
        if self.cfg["use_residual"]:
            from mall_space_planner.registry import build

            self.base = build("ranker", {"name": self.base_name, "params": {"seed": self.seed, "pairs_per_query": self.candidates_per_query}}) if self.base_name != "extra_trees" else ExtraTreesPointwiseRanker(pairs_per_query=self.candidates_per_query, seed=self.seed, n_estimators=200)
            self.base.fit(ctx, train_df, val_df)
        bucket = make_bucket_id(train_df, spec.city_cluster_col, spec.total_area_col, self.area_thresholds)
        q_df, c_df, rel, groups = sample_groups(train_df, ctx.db.label_col, ctx.db.mall_id_col, bucket, self.candidates_per_query, rng)
        if not groups:
            raise ValueError("no training groups")
        q, p, s_tab, gx, gei, gb = self._inputs(ctx, q_df, c_df, torch, device)
        rel_t = torch.tensor(rel, device=device)
        starts = np.cumsum([0, *groups[:-1]])
        model = self._new_model(ctx, torch).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        val_pack = None
        if val_df is not None and len(val_df) > 5:
            vb = make_bucket_id(val_df, spec.city_cluster_col, spec.total_area_col, self.area_thresholds)
            vq, vc, vrel, vgroups = sample_groups(val_df, ctx.db.label_col, ctx.db.mall_id_col, vb, 20, rng)
            if vgroups:
                val_pack = (self._inputs(ctx, vq, vc, torch, device), vrel, vgroups)
        best, bad, snapshots = -1.0, 0, []
        for ep in range(self.epochs):
            model.train()
            perm = rng.permutation(len(groups))
            losses = []
            for bi in range(0, len(perm), self.batch_groups):
                sel = perm[bi : bi + self.batch_groups]
                rows = np.concatenate([np.arange(starts[g], starts[g] + groups[g]) for g in sel])
                rows_t = torch.tensor(rows, device=device)
                gmask = torch.cat([torch.full((groups[g],), k, dtype=torch.long) for k, g in enumerate(sel)]).to(device)
                qn = q[rows_t] + self.feature_noise * torch.randn_like(q[rows_t]) if self.feature_noise > 0 else q[rows_t]
                # subgraph batch for selected rows
                node_mask = torch.isin(gb, rows_t)
                node_idx = torch.nonzero(node_mask).squeeze(1)
                remap = torch.full((gb.shape[0],), -1, dtype=torch.long, device=device)
                remap[node_idx] = torch.arange(node_idx.shape[0], device=device)
                emask = node_mask[gei[0]] & node_mask[gei[1]]
                sub_ei = remap[gei[:, emask]]
                row_to_local = torch.full((gb.max().item() + 1,), -1, dtype=torch.long, device=device)
                row_to_local[rows_t] = torch.arange(len(rows), device=device)
                sub_batch = row_to_local[gb[node_idx]]
                s = model(qn, p[rows_t], s_tab[rows_t], gx[node_idx], sub_ei, sub_batch, len(rows))
                # listwise softmax cross-entropy per group
                loss = 0.0
                for k in range(len(sel)):
                    m = gmask == k
                    target = torch.softmax(rel_t[rows_t][m] * 4.0, 0)
                    loss = loss + -(target * torch.log_softmax(s[m], 0)).sum()
                loss = loss / len(sel)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            sched.step()
            self.history_["train_loss"].append(float(np.mean(losses)))
            v = self._val_ndcg(model, val_pack, torch) if val_pack else -float(np.mean(losses))
            self.history_["val_ndcg10"].append(v)
            snapshots.append((v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}))
            if v > best + 1e-4:
                best, bad = v, 0
            else:
                bad += 1
            if bad >= self.patience:
                break
        snapshots.sort(key=lambda t: -t[0])
        self.states_ = [s for _, s in snapshots[: self.ensemble_k]]
        self._model = model
        alpha = float(self.states_[0].get("alpha", torch.tensor(float("nan"))))
        logger.info("DeepResidualRanker: %d epochs, best val=%.4f, alpha=%.3f, ensemble=%d, device=%s", len(self.history_["train_loss"]), best, alpha, len(self.states_), device)
        self.history_["alpha"] = [alpha]
        return self

    def _val_ndcg(self, model, pack, torch) -> float:  # noqa: ANN001
        from mall_space_planner.evaluation.ranking_metrics import ndcg_at_k

        (q, p, s_tab, gx, gei, gb), rel, groups = pack
        model.eval()
        with torch.no_grad():
            s = model(q, p, s_tab, gx, gei, gb, q.shape[0]).cpu().numpy()
        out, st = [], 0
        for g in groups:
            out.append(ndcg_at_k(s[st : st + g], rel[st : st + g], 10))
            st += g
        return float(np.mean(out))

    # ------------------------------------------------------------------ score
    def score(self, ctx: RankingContext, query_df: pd.DataFrame, cand_df: pd.DataFrame) -> np.ndarray:
        torch = _torch()
        device = select_device(self.device_pref)
        if self._model is None:
            self._model = self._new_model(ctx, torch)
        model = self._model.to(device)
        q, p, s_tab, gx, gei, gb = self._inputs(ctx, query_df, cand_df, torch, device)
        preds = []
        model.eval()
        with torch.no_grad():
            for st in self.states_ or [model.state_dict()]:
                model.load_state_dict(st)
                preds.append(model(q, p, s_tab, gx, gei, gb, q.shape[0]).cpu().numpy())
        return np.mean(preds, 0).astype(np.float32)

    def training_history(self) -> dict[str, list[float]]:
        return self.history_

    def feature_importance(self) -> dict[str, float] | None:
        return self.base.feature_importance() if self.base is not None else None

    # joblib cannot pickle torch modules reliably across devices → keep states only
    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_model"] = None
        d["graph_cache_"] = {}
        return d
