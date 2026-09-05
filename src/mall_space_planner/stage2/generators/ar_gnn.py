"""Autoregressive GNN topology expander (learned Stage-2 generator).

Task
----
Given a skeleton ``G_in`` and target size ``N_target``, add nodes one at a time until
``|V| = N_target``; skeleton edges are never removed (edge accuracy = 100 % by
construction, like the rule baselines, so the comparison is fair on the remaining metrics
and on **target-edge recall/precision**, which the rules cannot learn).

Model (pure PyTorch, no PyG)
----------------------------
At step *t* the current graph ``G_t`` is encoded by an L-layer GIN with node features
``[log1p degree, clustering, is_skeleton, is_new, normalised step t/N_target, layout one-hot]``
plus a graph-level context ``[|V_t|/N_target, avg_degree, layout one-hot]``. Two heads:

* **anchor head** – softmax over existing nodes: which node does the new node attach to;
* **second-edge head** – Bernoulli "add a second edge", and a softmax over existing nodes
  for its endpoint (conditioned on the anchor embedding), producing a 2-hop chord/loop.

Training: teacher forcing on a canonical **node ordering derived from the target graph**
(BFS from the skeleton; new nodes ordered by BFS distance then by degree). For each new
node *v* in that order the supervision is: anchor = its first target neighbour already in
``G_t``; second edge = any other already-present target neighbour (label 1) or none (0).
Loss = CE(anchor) + BCE(has_second) + CE(second endpoint | has_second).
Inference: greedy or temperature sampling with an optional metric-guided
**best-of-n re-ranking** using the same spec objective as ``search_expander`` (this makes
the AR model directly comparable: same search budget, learned vs. random proposals).

Small-sample tricks: node feature dropout, label smoothing, weight decay, early stopping
on validation anchor accuracy, snapshot ensembling (top-k checkpoints averaged in logit
space at inference).
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
NODE_DIM = 5 + len(_LAYOUTS)
CTX_DIM = 2 + len(_LAYOUTS)


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


# --------------------------------------------------------------------------- ordering / teacher forcing
def canonical_order(skeleton: TopologyGraph, target: TopologyGraph) -> list[str]:
    """New nodes of ``target`` ordered by BFS distance from the skeleton (ties: degree desc, name)."""
    g = to_networkx(target)
    sk = set(skeleton.nodes)
    dist: dict[str, int] = {n: 0 for n in sk if n in g}
    dq = deque(dist)
    while dq:
        u = dq.popleft()
        for v in g.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    new = [n for n in g.nodes if n not in sk]
    far = max(dist.values(), default=0) + 1
    return sorted(new, key=lambda n: (dist.get(n, far), -g.degree(n), n))


def teacher_steps(skeleton: TopologyGraph, target: TopologyGraph) -> list[tuple[list[str], list[tuple[str, str]], str, str | None]]:
    """Yield (present_nodes, present_edges, anchor, second_or_None) for each new node in canonical order."""
    tg = to_networkx(target)
    present = list(skeleton.nodes)
    edges = [e for e in skeleton.edges() if tg.has_edge(*e)] + [e for e in skeleton.edges() if not tg.has_edge(*e)]
    steps = []
    for v in canonical_order(skeleton, target):
        nbrs = [u for u in tg.neighbors(v) if u in set(present)]
        if not nbrs:  # disconnected in target → attach to a random present node later; skip as supervision
            present.append(v)
            continue
        nbrs.sort(key=lambda u: (-tg.degree(u), u))
        anchor, second = nbrs[0], (nbrs[1] if len(nbrs) > 1 else None)
        steps.append((list(present), list(edges), anchor, second))
        present.append(v)
        edges.append((anchor, v))
        if second:
            edges.append((second, v))
    return steps


# --------------------------------------------------------------------------- tensors
def state_tensors(present: list[str], edges: list[tuple[str, str]], skeleton_nodes: set[str], t_frac: float, layout_oh: np.ndarray, torch):  # noqa: ANN001, ANN202
    g = nx.Graph()
    g.add_nodes_from(present)
    g.add_edges_from(edges)
    idx = {n: i for i, n in enumerate(present)}
    n = len(present)
    deg = np.array([g.degree(v) for v in present], np.float32)
    clus = np.array([nx.clustering(g, v) for v in present], np.float32) if n > 2 else np.zeros(n, np.float32)
    is_sk = np.array([1.0 if v in skeleton_nodes else 0.0 for v in present], np.float32)
    x = np.concatenate([np.stack([np.log1p(deg), clus, is_sk, 1 - is_sk, np.full(n, t_frac, np.float32)], 1), np.tile(layout_oh, (n, 1))], 1)
    src = [idx[u] for u, v in edges] + [idx[v] for u, v in edges]
    dst = [idx[v] for u, v in edges] + [idx[u] for u, v in edges]
    ei = np.array([src, dst], np.int64) if src else np.zeros((2, 0), np.int64)
    ctx = np.concatenate([[t_frac, deg.mean() / 4.0 if n else 0.0], layout_oh]).astype(np.float32)
    return torch.tensor(x), torch.tensor(ei), torch.tensor(ctx)


def _build_model(torch):  # noqa: ANN001, ANN202
    nn = torch.nn

    class GIN(nn.Module):
        def __init__(self, d: int, dropout: float) -> None:
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
            self.eps = nn.Parameter(torch.zeros(1))
            self.norm = nn.LayerNorm(d)

        def forward(self, h, ei):  # noqa: ANN001, ANN202
            agg = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]]) if ei.shape[1] else torch.zeros_like(h)
            return self.norm(h + self.mlp((1 + self.eps) * h + agg))

    class ARExpander(nn.Module):
        def __init__(self, d_model: int, n_layers: int, dropout: float) -> None:
            super().__init__()
            d = d_model
            self.inp = nn.Linear(NODE_DIM, d)
            self.ctx = nn.Linear(CTX_DIM, d)
            self.layers = nn.ModuleList([GIN(d, dropout) for _ in range(n_layers)])
            self.anchor = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.has2 = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.second = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

        def encode(self, x, ei, ctx):  # noqa: ANN001, ANN202
            h = self.inp(x)
            for layer in self.layers:
                h = layer(h, ei)
            c = self.ctx(ctx)[None].expand(h.shape[0], -1)
            return h, c

        def forward(self, x, ei, ctx, anchor_idx=None):  # noqa: ANN001, ANN202
            h, c = self.encode(x, ei, ctx)
            a_logits = self.anchor(torch.cat([h, c], 1)).squeeze(-1)
            a = anchor_idx if anchor_idx is not None else int(a_logits.argmax())
            ha = h[a][None].expand(h.shape[0], -1)
            has2 = self.has2(torch.cat([h[a], c[0]])[None]).squeeze()
            s_logits = self.second(torch.cat([h, ha, c], 1)).squeeze(-1)
            mask = torch.zeros_like(s_logits)
            mask[a] = -30.0  # finite mask: anchor cannot be the second endpoint (label smoothing stays bounded)
            return a_logits, has2, s_logits + mask

    return ARExpander


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
    ) -> None:
        self.hp = dict(d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.lr, self.weight_decay, self.epochs, self.patience, self.label_smoothing = lr, weight_decay, epochs, patience, label_smoothing
        self.max_train_samples, self.val_fraction, self.ensemble_k = max_train_samples, val_fraction, ensemble_k
        self.temperature, self.best_of, self.w_aspl, self.device_pref, self.seed, self.label_style = temperature, best_of, w_aspl, device, seed, label_style
        self.states_: list[dict] = []
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_anchor_acc": [], "val_has2_acc": []}
        self._model = None
        if checkpoint:
            self.load(checkpoint)

    # ------------------------------------------------------------------ device/model
    def _device(self):  # noqa: ANN202
        torch = _torch()
        if self.device_pref != "auto":
            return torch.device(self.device_pref)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _model_or_new(self, torch):  # noqa: ANN001, ANN202
        if self._model is None:
            self._model = _build_model(torch)(**self.hp)
        return self._model

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
        logger.info("ARGNN: building teacher-forcing steps for %d train / %d val samples", len(train), len(val))
        tr_steps = [(s, st) for s in train for st in teacher_steps(s.skeleton, s.target)]
        va_steps = [(s, st) for s in val for st in teacher_steps(s.skeleton, s.target)]
        logger.info("ARGNN: %d train steps, %d val steps", len(tr_steps), len(va_steps))
        model = self._model_or_new(torch).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ce = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        ce_second = torch.nn.CrossEntropyLoss()  # no smoothing (masked anchor position)
        bce = torch.nn.BCEWithLogitsLoss()
        best, bad, snaps = -1.0, 0, []
        for ep in range(self.epochs):
            model.train()
            perm = rng.permutation(len(tr_steps))
            losses = []
            t0 = time.perf_counter()
            for i in perm:
                s, (present, edges, anchor, second) = tr_steps[i]
                x, ei, ctx = state_tensors(present, edges, set(s.skeleton.nodes), len(present) / max(1, s.target.num_nodes), _layout_onehot(s.layout_type), torch)
                x, ei, ctx = x.to(device), ei.to(device), ctx.to(device)
                a_idx = present.index(anchor)
                a_logits, has2, s_logits = model(x, ei, ctx, anchor_idx=a_idx)
                loss = ce(a_logits[None], torch.tensor([a_idx], device=device)) + bce(has2[None], torch.tensor([1.0 if second else 0.0], device=device))
                if second:
                    loss = loss + ce_second(s_logits[None], torch.tensor([present.index(second)], device=device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            acc_a, acc_2 = self._validate(model, va_steps, torch, device)
            self.history_["train_loss"].append(float(np.mean(losses)))
            self.history_["val_anchor_acc"].append(acc_a)
            self.history_["val_has2_acc"].append(acc_2)
            logger.info("ARGNN epoch %d: loss=%.4f val_anchor_acc=%.3f val_has2_acc=%.3f (%.0fs)", ep + 1, np.mean(losses), acc_a, acc_2, time.perf_counter() - t0)
            snaps.append((acc_a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
            if acc_a > best + 1e-4:
                best, bad = acc_a, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        snaps.sort(key=lambda t: -t[0])
        self.states_ = [s for _, s in snaps[: self.ensemble_k]]
        return self

    def _validate(self, model, steps, torch, device) -> tuple[float, float]:  # noqa: ANN001
        if not steps:
            return float("nan"), float("nan")
        model.eval()
        ok_a = ok_2 = 0
        with torch.no_grad():
            for s, (present, edges, anchor, second) in steps:
                x, ei, ctx = state_tensors(present, edges, set(s.skeleton.nodes), len(present) / max(1, s.target.num_nodes), _layout_onehot(s.layout_type), torch)
                a_logits, has2, _ = model(x.to(device), ei.to(device), ctx.to(device), anchor_idx=present.index(anchor))
                ok_a += int(int(a_logits.argmax()) == present.index(anchor))
                ok_2 += int((float(has2) > 0) == bool(second))
        return ok_a / len(steps), ok_2 / len(steps)

    # ------------------------------------------------------------------ generate
    def _sample_once(self, request: GenerationRequest, rng: np.random.RandomState, torch, device) -> TopologyGraph:  # noqa: ANN001
        sk = request.prototype.graph
        n_target = max(request.constraints.target_num_nodes or int(round(sk.num_nodes * 1.5)), sk.num_nodes)
        layout_oh = _layout_onehot(request.constraints.layout_type or request.prototype.layout_type)
        present, edges = list(sk.nodes), list(sk.edges())
        model = self._model_or_new(torch).to(device)
        model.eval()
        counter = len(present)
        with torch.no_grad():
            while len(present) < n_target:
                x, ei, ctx = state_tensors(present, edges, set(sk.nodes), len(present) / n_target, layout_oh, torch)
                x, ei, ctx = x.to(device), ei.to(device), ctx.to(device)
                a_logits_acc, has2_acc, s_acc = None, None, None
                for st in self.states_ or [model.state_dict()]:
                    model.load_state_dict(st)
                    a_l, _, _ = model(x, ei, ctx)
                    a_logits_acc = a_l if a_logits_acc is None else a_logits_acc + a_l
                a_probs = torch.softmax(a_logits_acc / (self.temperature * len(self.states_ or [1])), 0).cpu().numpy()
                a = int(rng.choice(len(present), p=a_probs / a_probs.sum())) if self.temperature > 0 else int(a_probs.argmax())
                for st in self.states_ or [model.state_dict()]:
                    model.load_state_dict(st)
                    _, h2, s_l = model(x, ei, ctx, anchor_idx=a)
                    has2_acc = h2 if has2_acc is None else has2_acc + h2
                    s_acc = s_l if s_acc is None else s_acc + s_l
                k = len(self.states_ or [1])
                p2 = float(torch.sigmoid(has2_acc / k))
                new = letter_label(counter) if self.label_style == "letters" else f"N{counter:03d}"
                while new in present:
                    counter += 1
                    new = letter_label(counter) if self.label_style == "letters" else f"N{counter:03d}"
                counter += 1
                edges.append((present[a], new))
                if rng.rand() < p2 and len(present) > 1:
                    s_probs = torch.softmax(s_acc / (self.temperature * k), 0).cpu().numpy()
                    s_probs[a] = 0
                    if s_probs.sum() > 0:
                        b = int(rng.choice(len(present), p=s_probs / s_probs.sum()))
                        edges.append((present[b], new))
                present.append(new)
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
        torch.save({"states": self.states_, "hp": self.hp, "history": self.history_, "seed": self.seed}, path / "ar_gnn.pt")
        (path / "meta.json").write_text(json.dumps({"hp": self.hp, "history": self.history_, "n_states": len(self.states_)}, indent=2), encoding="utf-8")
        return path

    def load(self, path: str | Path) -> ARGNNExpander:
        torch = _torch()
        blob = torch.load(Path(path) / "ar_gnn.pt", map_location="cpu", weights_only=False)
        self.hp, self.states_, self.history_ = blob["hp"], blob["states"], blob.get("history", self.history_)
        self._model = None
        return self
