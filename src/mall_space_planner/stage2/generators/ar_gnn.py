"""Autoregressive GNN topology expander (learned Stage-2 generator), v3.

v3 (real corpus from CSV graph files, see ``data/corpus_builder.py``)
-------------------------------------------------------------------
In the real corridor topologies a new key-point attaches to **1–4 already-present nodes**
(sample ``B000A0E928_1``: 25 new nodes, anchors per node = {1: 6, 2: 9, 3: 9, 4: 1}) and the target
is planar. The two heads "anchor" + "optional second edge" therefore become an **iterative
anchor / stop** policy: pick anchor a₁, then repeatedly decide *stop* or *pick another anchor*
(conditioned on the chosen set) up to ``max_anchors``. Training is teacher-forced over the
canonical anchor sequence with set likelihood at every sub-step; inference adds a **planarity
guard** (anchors that would make the graph non-planar are dropped) because corridor networks
are planar by construction.

Task
----
Given a skeleton ``G_in`` and target size ``N_target``, add nodes one at a time until
``|V| = N_target``; skeleton edges are never removed (edge accuracy = 100 % by
construction, like the rule baselines, so the comparison is fair on the remaining metrics
and on **target-edge recall/precision**, which the rules cannot learn).

What changed in v2 (after the first real-data run: val anchor acc 0.17, target-edge recall
= rule baseline)
------------------------------------------------------------------------------------------
* **Ordering = the corpus' own generation order.** New nodes in the ShareGPT corpus are
  labelled in the order they were created (``O, P, Q, ...`` form a corridor hanging off
  ``C, B, A``). Teacher forcing now follows label order (with deferral of nodes whose
  target neighbours are not present yet); BFS order is kept as an option. This also aligns
  the k-th generated node with the k-th target label, which is what the target-edge metric
  compares.
* **Structural node features** so that "which skeleton node gets the next corridor" is
  decidable: age of the node, is-last-added, BFS distance to the last added node, skeleton
  degree, number of new neighbours, leaf flag, random-walk structural encoding (RWSE).
* **Set likelihood for the anchor**: 31 % of steps have ≥2 valid anchors (the new node's
  target neighbours already present); the loss is ``-log Σ_{a∈valid} p(a)`` instead of an
  arbitrary single label. Same for the second endpoint.
* **Global readout in every head** (per-node scores are relative to the whole graph) and
  **batched teacher forcing** (``batch_steps`` graphs per backward pass, ~10× faster).

Inference: temperature sampling with an optional metric-guided best-of-n re-ranking
(same objective as ``search_expander`` → learned vs. random proposals at equal budget).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from mall_space_planner.data.sharegpt_adapter import ExpansionSample
from mall_space_planner.data.synthetic import letter_label
from mall_space_planner.registry import register
from mall_space_planner.schemas import LAYOUT_TYPES, LayoutType, TopologyGraph
from mall_space_planner.stage2.base import BaseTopologyGenerator, GenerationRequest
from mall_space_planner.topology.convert import from_networkx, to_networkx
from mall_space_planner.topology.metrics import aspl_deviation, density_deviation, node_deviation
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)

_LAYOUTS = list(LAYOUT_TYPES)
RWSE_K = 4
NODE_BASE = 13
NODE_DIM = NODE_BASE + RWSE_K + len(_LAYOUTS)
CTX_DIM = 3 + len(_LAYOUTS)
FEAT_VERSION = 3  # v3: multi-anchor heads (checkpoints from v2 must be retrained)
MAX_ANCHORS = 4


def _torch():  # noqa: ANN202
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ar_gnn requires torch") from exc
    return torch


def _layout_onehot(lt: LayoutType | str | None) -> np.ndarray:
    v = np.zeros(len(_LAYOUTS), np.float32)
    key = lt.value if isinstance(lt, LayoutType) else lt
    if key in _LAYOUTS:
        v[_LAYOUTS.index(key)] = 1.0
    return v


def label_key(n: str) -> tuple[int, str]:
    """Letter labels sort A..Z, AA..AZ, BA.. (length first); numeric-suffixed labels by number."""
    digits = "".join(ch for ch in n if ch.isdigit())
    return (len(n), n) if not digits or not n[0].isalpha() or n[:1].isdigit() else (10_000 + len(n), n)


# --------------------------------------------------------------------------- ordering / teacher forcing
def canonical_order(skeleton: TopologyGraph, target: TopologyGraph, order: str = "label") -> list[str]:
    """New nodes of ``target`` in generation order.

    ``label``: the corpus' own labelling order (creation order); nodes whose target neighbours are
    not present yet are deferred until one appears. ``bfs``: BFS distance from the skeleton
    (ties: degree desc, name).
    """
    g = to_networkx(target)
    sk = set(skeleton.nodes)
    new = [n for n in g.nodes if n not in sk]
    if order == "greedy":
        # most already-present target neighbours first (ties: label) -> most steps close loops
        present, rem, out_g = set(sk), set(new), []
        while rem:
            v = max(rem, key=lambda n: (sum(1 for u in g.neighbors(n) if u in present), tuple(-c for c in label_key(n)[:1]), n))
            out_g.append(v)
            present.add(v)
            rem.remove(v)
        return out_g
    if order == "bfs":
        dist: dict[str, int] = {n: 0 for n in sk if n in g}
        dq = deque(dist)
        while dq:
            u = dq.popleft()
            for v in g.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        far = max(dist.values(), default=0) + 1
        return sorted(new, key=lambda n: (dist.get(n, far), -g.degree(n), n))
    present = set(sk)
    out: list[str] = []
    pending: list[str] = []
    for v in sorted(new, key=label_key):
        pending.append(v)
        progressed = True
        while progressed:
            progressed = False
            for w in list(pending):
                if any(u in present for u in g.neighbors(w)):
                    out.append(w)
                    present.add(w)
                    pending.remove(w)
                    progressed = True
    # nodes never reachable from the skeleton (disconnected components) are appended last; they get
    # no anchor supervision (skipped in teacher_steps) but keep the label sequence intact
    return out + pending


def teacher_steps(skeleton: TopologyGraph, target: TopologyGraph, order: str = "label") -> list[dict[str, Any]]:
    """One supervision record per new node: state before adding it + the set of valid anchors."""
    tg = to_networkx(target)
    sk_nodes = list(skeleton.nodes)
    present = list(sk_nodes)
    added_at = {n: 0 for n in present}
    edges = list(skeleton.edges())
    steps = []
    for k, v in enumerate(canonical_order(skeleton, target, order), 1):
        pos = {n: i for i, n in enumerate(present)}
        valid = sorted((pos[u] for u in tg.neighbors(v) if u in pos), key=lambda i: (-tg.degree(present[i]), present[i]))
        if valid:
            steps.append({"present": list(present), "edges": list(edges), "added_at": dict(added_at), "k": k, "valid": valid})
        present.append(v)
        added_at[v] = k
        edges.extend((u, v) for u in tg.neighbors(v) if u in pos)
    return steps


# --------------------------------------------------------------------------- tensors
def _rwse(adj: np.ndarray, k: int) -> np.ndarray:
    deg = adj.sum(1, keepdims=True)
    p = np.divide(adj, deg, out=np.zeros_like(adj), where=deg > 0)
    out, m = [], np.eye(adj.shape[0], dtype=np.float32)
    for _ in range(k):
        m = m @ p
        out.append(np.diag(m))
    return np.stack(out, 1).astype(np.float32)


def state_features(present: list[str], edges: list[tuple[str, str]], skeleton_nodes: set[str], added_at: dict[str, int], k: int, n_target: int, layout_oh: np.ndarray, feature_set: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node features [n, NODE_DIM], edge index [2, 2E], context [CTX_DIM] for the state before step k.

    ``feature_set="basic"`` keeps only the v1 features (log degree, clustering, is_skeleton, is_new, t/N) and
    zeroes the structural block + RWSE — used as an ablation.
    """
    n = len(present)
    idx = {v: i for i, v in enumerate(present)}
    adj = np.zeros((n, n), np.float32)
    for u, v in edges:
        adj[idx[u], idx[v]] = adj[idx[v], idx[u]] = 1.0
    deg = adj.sum(1)
    is_sk = np.array([1.0 if v in skeleton_nodes else 0.0 for v in present], np.float32)
    sk_deg = (adj * is_sk[None, :]).sum(1) * is_sk  # degree within skeleton, 0 for new nodes
    new_nbrs = (adj * (1 - is_sk)[None, :]).sum(1)
    age = np.array([(k - added_at.get(v, 0)) for v in present], np.float32)
    last = [v for v in present if added_at.get(v, 0) == k - 1 and v not in skeleton_nodes] if k > 1 else []
    is_last = np.array([1.0 if v in last else 0.0 for v in present], np.float32)
    # BFS distance to the last-added node (capped); 1.0 if none
    dist_last = np.ones(n, np.float32)
    if last:
        d = {idx[last[0]]: 0}
        dq = deque(d)
        while dq:
            u = dq.popleft()
            if d[u] >= 4:
                continue
            for w in np.nonzero(adj[u])[0]:
                if int(w) not in d:
                    d[int(w)] = d[u] + 1
                    dq.append(int(w))
        for i, dd in d.items():
            dist_last[i] = dd / 4.0
    # clustering coefficient (triangles / possible)
    tri = np.einsum("ij,jk,ki->i", adj, adj, adj) / 2.0
    poss = deg * (deg - 1) / 2.0
    clus = np.divide(tri, poss, out=np.zeros_like(tri), where=poss > 0)
    t_frac = n / max(1, n_target)
    x = np.stack([
        np.log1p(deg), clus, is_sk, 1 - is_sk, np.full(n, t_frac, np.float32),
        np.minimum(age, 10) / 10.0 * (1 - is_sk) + is_sk, is_last, dist_last,
        np.log1p(sk_deg), np.log1p(new_nbrs), (deg == 1).astype(np.float32), deg / max(1.0, deg.max()),
        (deg == 0).astype(np.float32),
    ], 1).astype(np.float32)
    rw = _rwse(adj, RWSE_K)
    if feature_set == "basic":
        x[:, 5:] = 0.0
        rw = np.zeros_like(rw)
    x = np.concatenate([x, rw, np.tile(layout_oh, (n, 1))], 1)
    src = [idx[u] for u, v in edges] + [idx[v] for u, v in edges]
    dst = [idx[v] for u, v in edges] + [idx[u] for u, v in edges]
    ei = np.array([src, dst], np.int64) if src else np.zeros((2, 0), np.int64)
    ctx = np.concatenate([[t_frac, deg.mean() / 4.0 if n else 0.0, np.log1p(n_target) / 5.0], layout_oh]).astype(np.float32)
    return x, ei, ctx


# --------------------------------------------------------------------------- model
def _build_model(torch):  # noqa: ANN001, ANN202
    nn = torch.nn

    def seg_mean(h, batch, n_graphs):  # noqa: ANN001, ANN202
        s = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, batch, h)
        c = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype).index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype)).clamp(min=1)
        return s / c

    class GIN(nn.Module):
        def __init__(self, d: int, dropout: float) -> None:
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
            self.eps = nn.Parameter(torch.zeros(1))
            self.norm = nn.LayerNorm(d)

        def forward(self, h, ei, g_b):  # noqa: ANN001, ANN202
            agg = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]]) if ei.shape[1] else torch.zeros_like(h)
            return self.norm(h + self.mlp(torch.cat([(1 + self.eps) * h + agg, g_b], 1)))

    class ARExpander(nn.Module):
        def __init__(self, d_model: int, n_layers: int, dropout: float) -> None:
            super().__init__()
            d = d_model
            self.inp = nn.Sequential(nn.Linear(NODE_DIM, d), nn.GELU(), nn.Linear(d, d))
            self.ctx = nn.Linear(CTX_DIM, d)
            self.layers = nn.ModuleList([GIN(d, dropout) for _ in range(n_layers)])
            self.cnt = nn.Embedding(MAX_ANCHORS + 1, d)  # how many anchors already chosen
            # per-node score of "next anchor" given [node, chosen-set summary, graph, ctx, count]
            self.anchor = nn.Sequential(nn.Linear(5 * d, d), nn.GELU(), nn.Linear(d, 1))
            # stop / continue given [chosen-set summary, graph, ctx, count]
            self.stop = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))

        def encode(self, x, ei, ctx, batch, n_graphs):  # noqa: ANN001, ANN202
            h = self.inp(x)
            c = self.ctx(ctx)  # [B, d]
            for layer in self.layers:
                g = seg_mean(h, batch, n_graphs) + c
                h = layer(h, ei, g[batch])
            g = seg_mean(h, batch, n_graphs) + c
            return h, g, c

        def heads(self, h, g, c, node_src, sub_batch, sub_graph, chosen_sum, count):  # noqa: ANN001, ANN202
            """Score sub-steps. ``node_src``: replicated node -> original node index; ``sub_batch``: replicated
            node -> sub-step id; ``sub_graph``: sub-step -> graph id; ``chosen_sum`` [S, d]: mean h of chosen
            anchors (zeros if none); ``count`` [S]: number chosen so far. Returns (anchor logits [R], stop logits [S])."""
            ce = self.cnt(count)
            gs, cs = g[sub_graph], c[sub_graph]
            a = self.anchor(torch.cat([h[node_src], chosen_sum[sub_batch], gs[sub_batch], cs[sub_batch], ce[sub_batch]], 1)).squeeze(-1)
            s = self.stop(torch.cat([chosen_sum, gs, cs, ce], 1)).squeeze(-1)
            return a, s

        def forward(self, x, ei, ctx, batch, n_graphs, node_src, sub_batch, sub_graph, chosen_idx, chosen_sub, count):  # noqa: ANN001
            h, g, c = self.encode(x, ei, ctx, batch, n_graphs)
            S = int(sub_graph.shape[0])
            chosen_sum = torch.zeros(S, h.shape[1], device=h.device, dtype=h.dtype)
            if chosen_idx.numel():
                chosen_sum = chosen_sum.index_add(0, chosen_sub, h[chosen_idx])
                chosen_sum = chosen_sum / count.clamp(min=1).unsqueeze(1).to(h.dtype)
            return self.heads(h, g, c, node_src, sub_batch, sub_graph, chosen_sum, count)

    return ARExpander


def _pad(values, batch, n_graphs, fill=-1e9):  # noqa: ANN001, ANN202
    """[N] -> [B, L] assuming nodes of each graph are contiguous and graphs appear in order."""
    torch = _torch()
    counts = torch.bincount(batch, minlength=n_graphs)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(values.shape[0], device=values.device) - starts[batch]
    out = torch.full((n_graphs, int(counts.max().item())), fill, device=values.device, dtype=values.dtype)
    out[batch, pos] = values
    return out


def set_nll(logits, batch, n_graphs, valid_mask, smoothing: float = 0.0):  # noqa: ANN001, ANN202
    """-log Σ_{i∈valid} softmax(logits within graph)_i, mean over graphs (+ optional uniform smoothing)."""
    torch = _torch()
    pl = _pad(logits, batch, n_graphs)
    pv = _pad(valid_mask.to(logits.dtype), batch, n_graphs, fill=0.0) > 0.5
    lse_all = torch.logsumexp(pl, 1)
    lse_valid = torch.logsumexp(pl.masked_fill(~pv, -1e9), 1)
    nll = lse_all - lse_valid
    if smoothing > 0:
        n_nodes = (pl > -1e8).sum(1).clamp(min=1)
        uni = (lse_all[:, None] - pl).masked_fill(pl <= -1e8, 0.0).sum(1) / n_nodes
        nll = (1 - smoothing) * nll + smoothing * uni
    return nll.mean()


@register("generator", "ar_gnn")
class ARGNNExpander(BaseTopologyGenerator):
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        patience: int = 4,
        label_smoothing: float = 0.05,
        max_train_samples: int | None = None,
        val_fraction: float = 0.1,
        ensemble_k: int = 3,
        temperature: float = 0.7,
        best_of: int = 1,
        w_aspl: float = 1.5,
        device: str = "auto",
        seed: int = 42,
        checkpoint: str | None = None,
        label_style: str = "letters",
        order: str = "label",
        batch_steps: int = 64,
        second_from_valid_only: bool = True,
        loss_mode: str = "set",
        feature_set: str = "full",
        max_anchors: int = MAX_ANCHORS,
        planarity_guard: bool = True,
    ) -> None:
        """``loss_mode``: ``set`` = -log Σ_{valid} p (v2) | ``single`` = CE on the first valid anchor only (v1 ablation)."""
        self.hp = dict(d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.lr, self.weight_decay, self.epochs, self.patience, self.label_smoothing = lr, weight_decay, epochs, patience, label_smoothing
        self.max_train_samples, self.val_fraction, self.ensemble_k = max_train_samples, val_fraction, ensemble_k
        self.temperature, self.best_of, self.w_aspl, self.device_pref, self.seed, self.label_style = temperature, best_of, w_aspl, device, seed, label_style
        self.order, self.batch_steps, self.second_from_valid_only = order, batch_steps, second_from_valid_only
        self.loss_mode, self.feature_set = loss_mode, feature_set
        self.max_anchors, self.planarity_guard = min(max_anchors, MAX_ANCHORS), planarity_guard
        self.states_: list[dict] = []
        # val_has2_acc is kept for backwards-compatible plots: it is now the stop/continue accuracy
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_anchor_acc": [], "val_has2_acc": [], "val_anchor_top3": [], "val_count_acc": []}
        self._model = None
        self._ens: list[Any] = []
        if checkpoint:
            self.load(checkpoint)

    # ------------------------------------------------------------------ device/model
    def _device(self):  # noqa: ANN202
        from mall_space_planner.stage1.rankers.deep_ranker import select_device  # same policy: auto -> cuda|cpu, mps opt-in

        return select_device(self.device_pref)

    def _model_or_new(self, torch):  # noqa: ANN001, ANN202
        if self._model is None:
            self._model = _build_model(torch)(**self.hp)
        return self._model

    # ------------------------------------------------------------------ data
    def _materialise(self, sample: ExpansionSample, step: dict[str, Any]) -> dict[str, Any]:
        x, ei, ctx = state_features(step["present"], step["edges"], set(sample.skeleton.nodes), step["added_at"], step["k"], sample.target.num_nodes, _layout_onehot(sample.layout_type), self.feature_set)
        return {"x": x, "ei": ei, "ctx": ctx, "valid": np.array(step["valid"], np.int64), "n": len(step["present"])}

    def _collate(self, items: list[dict[str, Any]], torch, device):  # noqa: ANN001, ANN202
        """Batch of teacher steps -> graph tensors + sub-step tensors.

        For a step with ordered valid anchors V (|V| = m, truncated to ``max_anchors``) we create sub-steps
        j = 0..m: chosen = V[:j]; at j < m the correct move is "pick any of V[j:]" (set likelihood, stop = 0);
        at j = m the correct move is "stop". ``loss_mode == "single"`` keeps only j = 0 with V[:1] (v1 ablation).
        """
        xs, eis, ctxs, batch, off = [], [], [], [], 0
        node_src, sub_batch, sub_graph, chosen_idx, chosen_sub, count, tgt_mask, stop_y = [], [], [], [], [], [], [], []
        s_id = 0
        for b, it in enumerate(items):
            n = it["n"]
            xs.append(it["x"]); eis.append(it["ei"] + off); ctxs.append(it["ctx"]); batch.append(np.full(n, b, np.int64))
            V = [int(v) for v in it["valid"]][: self.max_anchors]
            if self.loss_mode == "single":
                V = V[:1]
            subs = range(len(V) + 1) if self.loss_mode != "single" else range(1)
            for j in subs:
                node_src.append(np.arange(n) + off); sub_batch.append(np.full(n, s_id, np.int64)); sub_graph.append(b)
                for a in V[:j]:
                    chosen_idx.append(off + a); chosen_sub.append(s_id)
                count.append(j)
                m = np.zeros(n, np.float32)
                if j < len(V):
                    m[V[j:]] = 1.0
                    stop_y.append(0.0)
                else:
                    stop_y.append(1.0)
                tgt_mask.append(m)
                s_id += 1
            off += n
        T = lambda a, dt=None: torch.tensor(np.concatenate(a) if isinstance(a, list) and len(a) and isinstance(a[0], np.ndarray) else np.asarray(a), dtype=dt, device=device)  # noqa: E731
        return {
            "x": T(xs), "ei": torch.tensor(np.concatenate(eis, 1), device=device), "ctx": torch.tensor(np.stack(ctxs), device=device), "batch": T(batch), "B": len(items),
            "node_src": T(node_src, torch.long), "sub_batch": T(sub_batch, torch.long), "sub_graph": torch.tensor(np.asarray(sub_graph, np.int64), device=device),
            "chosen_idx": torch.tensor(np.asarray(chosen_idx, np.int64), device=device), "chosen_sub": torch.tensor(np.asarray(chosen_sub, np.int64), device=device),
            "count": torch.tensor(np.asarray(count, np.int64), device=device), "tgt_mask": T(tgt_mask), "stop_y": torch.tensor(np.asarray(stop_y, np.float32), device=device),
        }

    def _forward(self, model, d):  # noqa: ANN001, ANN202
        return model(d["x"], d["ei"], d["ctx"], d["batch"], d["B"], d["node_src"], d["sub_batch"], d["sub_graph"], d["chosen_idx"], d["chosen_sub"], d["count"])

    # ------------------------------------------------------------------ fit
    def fit(self, samples: list[ExpansionSample]) -> ARGNNExpander:  # type: ignore[override]
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        device = self._device()
        samples = [s for s in samples if s.target.num_nodes > s.skeleton.num_nodes]
        if self.max_train_samples:
            samples = samples[: self.max_train_samples]
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * self.val_fraction))
        val, train = samples[:n_val], samples[n_val:]
        logger.info("ARGNN: building teacher-forcing steps (order=%s) for %d train / %d val samples", self.order, len(train), len(val))
        t0 = time.perf_counter()
        tr = [self._materialise(s, st) for s in train for st in teacher_steps(s.skeleton, s.target, self.order)]
        va = [self._materialise(s, st) for s in val for st in teacher_steps(s.skeleton, s.target, self.order)]
        logger.info("ARGNN: %d train steps, %d val steps (features %.0fs)", len(tr), len(va), time.perf_counter() - t0)
        model = self._model_or_new(torch).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=max(1, self.epochs * ((len(tr) + self.batch_steps - 1) // self.batch_steps)), pct_start=0.15)
        bce = torch.nn.BCEWithLogitsLoss()
        best, bad, snaps = -1.0, 0, []
        for ep in range(self.epochs):
            model.train()
            perm = rng.permutation(len(tr))
            losses = []
            t0 = time.perf_counter()
            for bi in range(0, len(perm), self.batch_steps):
                items = [tr[i] for i in perm[bi : bi + self.batch_steps]]
                d = self._collate(items, torch, device)
                a_logits, s_logits = self._forward(model, d)
                S = int(d["sub_graph"].shape[0])
                pick = d["stop_y"] < 0.5  # sub-steps where an anchor must be picked
                loss = bce(s_logits, d["stop_y"]) if self.loss_mode != "single" else torch.zeros((), device=device)
                if bool(pick.any()):
                    # already-chosen anchors are not candidates again
                    tm = d["tgt_mask"].clone()
                    al = a_logits.clone()
                    if d["chosen_idx"].numel():
                        # map (chosen node, sub-step) -> replicated row: rows are contiguous per sub-step in node order
                        starts = torch.cumsum(torch.bincount(d["sub_batch"], minlength=S), 0) - torch.bincount(d["sub_batch"], minlength=S)
                        g_off = torch.cumsum(torch.bincount(d["batch"], minlength=d["B"]), 0) - torch.bincount(d["batch"], minlength=d["B"])
                        rows = starts[d["chosen_sub"]] + (d["chosen_idx"] - g_off[d["sub_graph"][d["chosen_sub"]]])
                        al[rows] = -30.0
                    keep = pick[d["sub_batch"]]
                    remap = torch.cumsum(pick.long(), 0) - 1
                    loss = loss + set_nll(al[keep], remap[d["sub_batch"][keep]], int(pick.sum()), tm[keep] > 0.5, self.label_smoothing)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            acc_a, top3, acc_2 = self._validate(model, va, torch, device)
            self.history_["train_loss"].append(float(np.mean(losses)))
            self.history_["val_anchor_acc"].append(acc_a)
            self.history_["val_anchor_top3"].append(top3)
            self.history_["val_has2_acc"].append(acc_2)
            logger.info("ARGNN epoch %d/%d: loss=%.4f val_anchor_acc=%.3f top3=%.3f val_has2_acc=%.3f (%.0fs)", ep + 1, self.epochs, np.mean(losses), acc_a, top3, acc_2, time.perf_counter() - t0)
            snaps.append((acc_a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
            if acc_a > best + 1e-4:
                best, bad = acc_a, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        snaps.sort(key=lambda t: -t[0])
        self.states_ = [s for _, s in snaps[: self.ensemble_k]]
        self._ens = []
        return self

    def _validate(self, model, items, torch, device) -> tuple[float, float, float]:  # noqa: ANN001
        """(first-anchor acc, first-anchor top-3, stop/continue acc over all sub-steps)."""
        if not items:
            return float("nan"), float("nan"), float("nan")
        model.eval()
        ok_a = ok_3 = ok_s = n_s = 0
        with torch.no_grad():
            for bi in range(0, len(items), 128):
                chunk = items[bi : bi + 128]
                d = self._collate(chunk, torch, device)
                a_logits, s_logits = self._forward(model, d)
                first = d["count"] == 0  # sub-step 0 of every step
                sel = first[d["sub_batch"]]
                S0 = int(first.sum())
                remap = torch.cumsum(first.long(), 0) - 1
                pl = _pad(a_logits[sel], remap[d["sub_batch"][sel]], S0)
                pv = _pad(d["tgt_mask"][sel], remap[d["sub_batch"][sel]], S0, fill=0.0) > 0.5
                top = pl.topk(min(3, pl.shape[1]), 1).indices
                hit = pv.gather(1, top)
                ok_a += int(hit[:, 0].sum()); ok_3 += int(hit.any(1).sum())
                if self.loss_mode != "single":
                    ok_s += int(((s_logits > 0).float() == d["stop_y"]).sum()); n_s += int(d["stop_y"].numel())
        return ok_a / len(items), ok_3 / len(items), (ok_s / n_s if n_s else float("nan"))

    # ------------------------------------------------------------------ generate
    def _ensemble(self, torch, device):  # noqa: ANN001, ANN202
        if not self._ens:
            states = self.states_ or [self._model_or_new(torch).state_dict()]
            for st in states:
                m = _build_model(torch)(**self.hp)
                m.load_state_dict(st)
                self._ens.append(m.to(device).eval())
        return self._ens

    def _sample_once(self, request: GenerationRequest, rng: np.random.RandomState, torch, device) -> TopologyGraph:  # noqa: ANN001
        sk = request.prototype.graph
        n_target = max(request.constraints.target_num_nodes or int(round(sk.num_nodes * 1.5)), sk.num_nodes)
        layout_oh = _layout_onehot(request.constraints.layout_type or request.prototype.layout_type)
        sk_nodes = set(sk.nodes)
        present, edges = list(sk.nodes), list(sk.edges())
        added_at = {n: 0 for n in present}
        models = self._ensemble(torch, device)
        k_ens = len(models)
        counter = len(present)
        k = 1
        tau = max(self.temperature, 1e-3)
        g_live = nx.Graph()
        g_live.add_nodes_from(present)
        g_live.add_edges_from(edges)
        with torch.no_grad():
            while len(present) < n_target:
                n = len(present)
                x, ei, ctx = state_features(present, edges, sk_nodes, added_at, k, n_target, layout_oh, self.feature_set)
                x_t, ei_t, ctx_t = torch.tensor(x, device=device), torch.tensor(ei, device=device), torch.tensor(ctx[None], device=device)
                batch = torch.zeros(n, dtype=torch.long, device=device)
                enc = [m.encode(x_t, ei_t, ctx_t, batch, 1) for m in models]  # encode once per step
                node_src = torch.arange(n, device=device)
                sub_batch = torch.zeros(n, dtype=torch.long, device=device)
                sub_graph = torch.zeros(1, dtype=torch.long, device=device)
                new = letter_label(counter) if self.label_style == "letters" else f"N{counter:03d}"
                while new in present:
                    counter += 1
                    new = letter_label(counter) if self.label_style == "letters" else f"N{counter:03d}"
                counter += 1
                chosen: list[int] = []
                banned: set[int] = set()
                while len(chosen) < self.max_anchors:
                    cnt = torch.tensor([len(chosen)], device=device)
                    a_acc, s_acc = 0.0, 0.0
                    for m, (h, g, c) in zip(models, enc, strict=True):
                        cs = h[chosen].mean(0, keepdim=True) if chosen else torch.zeros(1, h.shape[1], device=device, dtype=h.dtype)
                        a_l, s_l = m.heads(h, g, c, node_src, sub_batch, sub_graph, cs, cnt)
                        a_acc, s_acc = a_acc + a_l, s_acc + s_l
                    if chosen:  # stop / continue decision (first anchor is mandatory)
                        p_stop = float(torch.sigmoid(s_acc / k_ens))
                        stop = (rng.rand() < p_stop) if self.temperature > 0 else (p_stop > 0.5)
                        if stop:
                            break
                    probs = torch.softmax(a_acc / (tau * k_ens), 0).cpu().numpy().astype(float)
                    for i in chosen:
                        probs[i] = 0.0
                    for i in banned:
                        probs[i] = 0.0
                    if probs.sum() <= 0:
                        break
                    probs /= probs.sum()
                    a = int(rng.choice(n, p=probs)) if self.temperature > 0 else int(probs.argmax())
                    if self.planarity_guard and chosen:
                        # corridor networks are planar: reject an anchor that would break planarity
                        g_live.add_node(new)
                        g_live.add_edges_from((present[i], new) for i in [*chosen, a])
                        ok = nx.check_planarity(g_live)[0]
                        g_live.remove_node(new)
                        if not ok:
                            banned.add(a)
                            if len(banned) >= 3:
                                break
                            continue
                    chosen.append(a)
                new_edges = [(present[i], new) for i in chosen]
                edges.extend(new_edges)
                present.append(new)
                added_at[new] = k
                g_live.add_node(new)
                g_live.add_edges_from(new_edges)
                k += 1
        out = from_networkx(g_live)
        out.node_types = {n: sk.node_types.get(n, "M") for n in out.nodes}
        out.positions = dict(sk.positions)
        return out

    def generate(self, request: GenerationRequest, seed: int) -> TopologyGraph:
        torch = _torch()
        device = self._device()
        rng = np.random.RandomState(seed)
        sk = request.prototype.graph
        n_target = max(request.constraints.target_num_nodes or int(round(sk.num_nodes * 1.5)), sk.num_nodes)
        best, best_obj = None, float("inf")
        for _ in range(max(1, self.best_of)):
            g = self._sample_once(request, rng, torch, device)
            if self.best_of <= 1:
                return g
            comps = nx.number_connected_components(to_networkx(g))
            obj = node_deviation(g, n_target) + density_deviation(sk, g, n_target) + self.w_aspl * aspl_deviation(sk, g, n_target) + 100 * max(0, comps - 1)
            if obj < best_obj:
                best, best_obj = g, obj
        return best  # type: ignore[return-value]

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        torch = _torch()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"hp": self.hp, "history": self.history_, "n_states": len(self.states_), "order": self.order, "feat_version": FEAT_VERSION, "loss_mode": self.loss_mode, "feature_set": self.feature_set, "max_anchors": self.max_anchors}
        torch.save({"states": self.states_, "seed": self.seed, **meta}, path / "ar_gnn.pt")
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def load(self, path: str | Path) -> ARGNNExpander:
        torch = _torch()
        blob = torch.load(Path(path) / "ar_gnn.pt", map_location="cpu", weights_only=False)
        if blob.get("feat_version", 1) != FEAT_VERSION:
            raise ValueError(f"checkpoint {path} was trained with feature version {blob.get('feat_version', 1)}; retrain with scripts/train_stage2.py")
        self.hp, self.states_, self.history_ = blob["hp"], blob["states"], blob.get("history", self.history_)
        self.order = blob.get("order", self.order)
        self.feature_set = blob.get("feature_set", self.feature_set)
        self.loss_mode = blob.get("loss_mode", self.loss_mode)
        self.max_anchors = int(blob.get("max_anchors", self.max_anchors))
        self._model, self._ens = None, []
        return self
