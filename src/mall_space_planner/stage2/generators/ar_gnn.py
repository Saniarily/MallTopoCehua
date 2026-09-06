"""Autoregressive GNN topology expander (learned Stage-2 generator), v2.

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
FEAT_VERSION = 2


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
            self.anchor = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.has2 = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.second = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))

        def encode(self, x, ei, ctx, batch, n_graphs):  # noqa: ANN001, ANN202
            h = self.inp(x)
            c = self.ctx(ctx)  # [B, d]
            for layer in self.layers:
                g = seg_mean(h, batch, n_graphs) + c
                h = layer(h, ei, g[batch])
            g = seg_mean(h, batch, n_graphs) + c
            return h, g

        def forward(self, x, ei, ctx, batch, n_graphs, anchor_idx=None):  # noqa: ANN001, ANN202
            """anchor_idx: [B] global node indices (teacher forcing) or None (use argmax per graph)."""
            h, g = self.encode(x, ei, ctx, batch, n_graphs)
            gb = g[batch]
            a_logits = self.anchor(torch.cat([h, gb, self.ctx(ctx)[batch]], 1)).squeeze(-1)
            if anchor_idx is None:
                # per-graph argmax (used at inference with B=1)
                anchor_idx = torch.stack([(a_logits.masked_fill(batch != b, -1e9)).argmax() for b in range(n_graphs)])
            ha = h[anchor_idx]  # [B, d]
            has2 = self.has2(torch.cat([ha, g, self.ctx(ctx)], 1)).squeeze(-1)  # [B]
            s_logits = self.second(torch.cat([h, ha[batch], gb, self.ctx(ctx)[batch]], 1)).squeeze(-1)
            mask = torch.zeros_like(s_logits)
            mask[anchor_idx] = -30.0
            return a_logits, has2, s_logits + mask

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
    ) -> None:
        """``loss_mode``: ``set`` = -log Σ_{valid} p (v2) | ``single`` = CE on the first valid anchor only (v1 ablation)."""
        self.hp = dict(d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.lr, self.weight_decay, self.epochs, self.patience, self.label_smoothing = lr, weight_decay, epochs, patience, label_smoothing
        self.max_train_samples, self.val_fraction, self.ensemble_k = max_train_samples, val_fraction, ensemble_k
        self.temperature, self.best_of, self.w_aspl, self.device_pref, self.seed, self.label_style = temperature, best_of, w_aspl, device, seed, label_style
        self.order, self.batch_steps, self.second_from_valid_only = order, batch_steps, second_from_valid_only
        self.loss_mode, self.feature_set = loss_mode, feature_set
        self.states_: list[dict] = []
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_anchor_acc": [], "val_has2_acc": [], "val_anchor_top3": []}
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
        xs, eis, ctxs, batch, valid, anchor, off = [], [], [], [], [], [], 0
        for b, it in enumerate(items):
            xs.append(it["x"])
            eis.append(it["ei"] + off)
            ctxs.append(it["ctx"])
            batch.append(np.full(it["n"], b, np.int64))
            vm = np.zeros(it["n"], np.float32)
            vm[it["valid"]] = 1.0
            valid.append(vm)
            anchor.append(off + int(it["valid"][0]))
            off += it["n"]
        t = lambda a, dt=None: torch.tensor(np.concatenate(a) if isinstance(a, list) else a, dtype=dt, device=device)  # noqa: E731
        return t(xs), torch.tensor(np.concatenate(eis, 1), device=device), torch.tensor(np.stack(ctxs), device=device), t(batch), t(valid), torch.tensor(anchor, device=device)

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
                x, ei, ctx, batch, vmask, a_idx = self._collate(items, torch, device)
                B = len(items)
                a_logits, has2, s_logits = model(x, ei, ctx, batch, B, anchor_idx=a_idx)
                n_valid = torch.tensor([len(it["valid"]) for it in items], device=device)
                if self.loss_mode == "single":  # v1-style: only the first valid anchor counts as correct
                    single = torch.zeros_like(vmask)
                    single[a_idx] = 1.0
                    loss = set_nll(a_logits, batch, B, single > 0.5, self.label_smoothing) + bce(has2, (n_valid >= 2).float())
                else:
                    loss = set_nll(a_logits, batch, B, vmask > 0.5, self.label_smoothing) + bce(has2, (n_valid >= 2).float())
                two = n_valid >= 2
                if bool(two.any()):
                    v2 = vmask.clone()
                    v2[a_idx] = 0.0  # anchor is not a candidate for the second endpoint
                    keep = two[batch]
                    # restrict to graphs with a second edge: recompute compact batch ids
                    remap = torch.cumsum(two.long(), 0) - 1
                    loss = loss + set_nll(s_logits[keep], remap[batch[keep]], int(two.sum()), v2[keep] > 0.5)
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
        if not items:
            return float("nan"), float("nan"), float("nan")
        model.eval()
        ok_a = ok_3 = ok_2 = 0
        with torch.no_grad():
            for bi in range(0, len(items), 256):
                chunk = items[bi : bi + 256]
                x, ei, ctx, batch, vmask, a_idx = self._collate(chunk, torch, device)
                B = len(chunk)
                a_logits, has2, _ = model(x, ei, ctx, batch, B, anchor_idx=a_idx)
                pl, pv = _pad(a_logits, batch, B), _pad(vmask, batch, B, fill=0.0) > 0.5
                top = pl.topk(min(3, pl.shape[1]), 1).indices
                hit = pv.gather(1, top)
                ok_a += int(hit[:, 0].sum())
                ok_3 += int(hit.any(1).sum())
                n_valid = torch.tensor([len(it["valid"]) for it in chunk], device=device)
                ok_2 += int(((has2 > 0) == (n_valid >= 2)).sum())
        return ok_a / len(items), ok_3 / len(items), ok_2 / len(items)

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
        with torch.no_grad():
            while len(present) < n_target:
                x, ei, ctx = state_features(present, edges, sk_nodes, added_at, k, n_target, layout_oh, self.feature_set)
                x_t, ei_t, ctx_t = torch.tensor(x, device=device), torch.tensor(ei, device=device), torch.tensor(ctx[None], device=device)
                batch = torch.zeros(len(present), dtype=torch.long, device=device)
                a_acc = sum(m(x_t, ei_t, ctx_t, batch, 1, anchor_idx=torch.zeros(1, dtype=torch.long, device=device))[0] for m in models)
                a_probs = torch.softmax(a_acc / (max(self.temperature, 1e-3) * k_ens), 0).cpu().numpy()
                a = int(rng.choice(len(present), p=a_probs / a_probs.sum())) if self.temperature > 0 else int(a_probs.argmax())
                a_t = torch.tensor([a], device=device)
                h2_acc, s_acc = 0.0, 0.0
                for m in models:
                    _, h2, s_l = m(x_t, ei_t, ctx_t, batch, 1, anchor_idx=a_t)
                    h2_acc, s_acc = h2_acc + h2, s_acc + s_l
                p2 = float(torch.sigmoid(h2_acc / k_ens))
                new = letter_label(counter) if self.label_style == "letters" else f"N{counter:03d}"
                while new in present:
                    counter += 1
                    new = letter_label(counter) if self.label_style == "letters" else f"N{counter:03d}"
                counter += 1
                new_edges = [(present[a], new)]
                if rng.rand() < p2 and len(present) > 1:
                    s_probs = torch.softmax(s_acc / (max(self.temperature, 1e-3) * k_ens), 0).cpu().numpy()
                    s_probs[a] = 0
                    if s_probs.sum() > 0:
                        b = int(rng.choice(len(present), p=s_probs / s_probs.sum())) if self.temperature > 0 else int(s_probs.argmax())
                        new_edges.append((present[b], new))
                edges.extend(new_edges)
                present.append(new)
                added_at[new] = k
                k += 1
        g = nx.Graph()
        g.add_nodes_from(present)
        g.add_edges_from(edges)
        out = from_networkx(g)
        out.node_types = {n: sk.node_types.get(n, "M") for n in out.nodes}
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
        meta = {"hp": self.hp, "history": self.history_, "n_states": len(self.states_), "order": self.order, "feat_version": FEAT_VERSION, "loss_mode": self.loss_mode, "feature_set": self.feature_set}
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
        self._model, self._ens = None, []
        return self
