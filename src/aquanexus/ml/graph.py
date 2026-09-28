"""Graph neural networks over the monitoring-station network (Phase 8).

**BLOCKED by network size and missing topology.** ``docs/ML_ROADMAP.md``
Phase 8 and ``docs/DATA_LIMITATIONS.md`` establish why: only 4 real Ayase
stations (5 Naka) exist as potential graph nodes, and no upstream/downstream
edge list between *monitoring stations* is encoded anywhere in this
repository. `river_station` in the HEC-RAS data (`hecras/geometry.py`) orders
*cross-sections* along one continuous reach for hydraulic interpolation - a
different graph than the monitoring network, and never joined to a
monitoring station's identity anywhere in `data/dataset.py`. A GCN/GAT over a
single-digit-node graph with no real edges would produce a number that is an
artifact of a toy graph, not evidence about river network structure - exactly
what the spec this project follows says not to report.

Everything below is generic graph-neural-network machinery: a dense-graph GCN
layer (Kipf & Welling 2017's renormalization trick) and a dense-graph GAT
layer (Velickovic et al. 2018's attention mechanism), both hand-rolled in
plain PyTorch rather than adding `torch_geometric` as a dependency - at 4-10
nodes, a dense adjacency matrix needs nothing a sparse-graph library
provides, and `torch_geometric`'s Windows wheel compatibility is a real,
avoidable risk for a graph this small. Tested against synthetic small graphs
only (``tests/test_graph.py``), so that if real station topology is sourced
later, the mechanism does not need to be built from scratch.
:func:`require_station_topology` is the pipeline's entry guard - see
``scripts/phase8_gnn_feasibility.py`` for it raising against the real
station list, the concrete evidence behind the "blocked" status above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquanexus.logger import get_logger

log = get_logger("ml.graph")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PyTorch is required for aquanexus.ml.graph. Install the 'deep' "
            "optional extra: pip install 'aquanexus[deep]'"
        ) from exc
    return torch


@dataclass
class StationGraph:
    """A small station network: node order, adjacency, and node features."""

    node_names: list[str]
    adjacency: np.ndarray  # (n, n), 1.0 where an edge exists, symmetric or directed
    features: np.ndarray  # (n, n_features)

    def __post_init__(self):
        n = len(self.node_names)
        if self.adjacency.shape != (n, n):
            raise ValueError(f"adjacency must be ({n}, {n}), got {self.adjacency.shape}")
        if self.features.shape[0] != n:
            raise ValueError(f"features must have {n} row(s), got {self.features.shape[0]}")


def build_station_graph(node_names: list[str], edges: list[tuple[str, str]],
                        features: np.ndarray, directed: bool = True) -> StationGraph:
    """Build a :class:`StationGraph` from an explicit edge list.

    ``edges`` are ``(upstream, downstream)`` pairs by name. Nodes are never
    connected merely because they are geographically close - every edge here
    must come from an actual hydrological relationship the caller supplies,
    per the spec this project follows ("do not connect nodes merely because
    they are geographically close unless there is a scientific
    justification"). If ``directed`` is False, each edge is mirrored.
    """
    index = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for upstream, downstream in edges:
        if upstream not in index or downstream not in index:
            raise ValueError(f"edge ({upstream!r}, {downstream!r}) references an unknown node")
        adjacency[index[upstream], index[downstream]] = 1.0
        if not directed:
            adjacency[index[downstream], index[upstream]] = 1.0
    return StationGraph(node_names=list(node_names), adjacency=adjacency, features=features)


def _normalized_adjacency(adjacency) -> np.ndarray:
    """Kipf & Welling's renormalization trick: ``D^-1/2 (A + I) D^-1/2``."""
    n = adjacency.shape[0]
    a_hat = adjacency + np.eye(n, dtype=adjacency.dtype)
    degree = a_hat.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
    return (d_inv_sqrt[:, None] * a_hat) * d_inv_sqrt[None, :]


class GCNLayer:
    """One dense graph-convolution layer: ``sigma(A_hat @ X @ W)``.

    ``A_hat`` (the renormalized adjacency) is fixed, precomputed once per
    graph - it is data (the network topology), not a learned parameter.
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        torch = _require_torch()
        from torch import nn

        gen = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight, generator=gen)

    def __call__(self, a_hat, x):
        torch = _require_torch()
        return torch.nn.functional.relu(self.linear(a_hat @ x))

    def parameters(self):
        return self.linear.parameters()


class GATLayer:
    """One dense graph-attention layer (single head), masked by adjacency.

    Attention weight from node j to node i: ``softmax_j(LeakyReLU(a^T
    [Wh_i || Wh_j]))`` over j in i's neighborhood (including i itself, via a
    self-loop added to the mask) - the mechanism from Velickovic et al. 2018,
    implemented densely since the graphs this module ever runs on (synthetic
    tests, or a real station network too small to need sparsity) fit
    trivially in a dense adjacency matrix.
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        torch = _require_torch()
        from torch import nn

        gen = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Linear(2 * out_features, 1, bias=False)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight, generator=gen)
            nn.init.xavier_uniform_(self.attn.weight, generator=gen)

    def __call__(self, adjacency, x):
        torch = _require_torch()
        n = x.shape[0]
        h = self.linear(x)  # (n, out_features)

        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        concatenated = torch.cat([h_i, h_j], dim=-1)  # (n, n, 2*out_features)
        scores = torch.nn.functional.leaky_relu(self.attn(concatenated).squeeze(-1), 0.2)

        mask = (adjacency + torch.eye(n, dtype=adjacency.dtype)) > 0
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        return torch.nn.functional.elu(attention @ h)

    def parameters(self):
        return list(self.linear.parameters()) + list(self.attn.parameters())


class _DenseGraphModel:
    """Shared fit/predict machinery for :class:`GCNModel` and :class:`GATModel`.

    Transductive node regression: all nodes' features and the fixed graph are
    visible at every forward pass; only the *loss* is masked to the training
    nodes, so a held-out node's own label never influences training, but its
    neighbors' messages still reach it exactly as they would in a real
    deployment (`tests/test_graph.py` verifies this is genuinely different
    from an isolated per-node MLP with no message passing).
    """

    def __init__(self, layer_cls, hidden_size: int = 8, learning_rate: float = 1e-2,
                weight_decay: float = 1e-3, max_epochs: int = 300, patience: int = 30,
                seed: int = 42):
        self.layer_cls = layer_cls
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.train_loss: list[float] = []

    def fit(self, graph: StationGraph, y: np.ndarray, train_mask: np.ndarray):
        torch = _require_torch()
        from aquanexus.ml.deep import set_seed

        set_seed(self.seed)
        n_features = graph.features.shape[1]
        self.layer1 = self.layer_cls(n_features, self.hidden_size, seed=self.seed)
        self.layer2 = self.layer_cls(self.hidden_size, 1, seed=self.seed + 1)

        x = torch.from_numpy(graph.features.astype(np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        mask = torch.from_numpy(np.asarray(train_mask, dtype=bool))

        if self.layer_cls is GCNLayer:
            self._a = torch.from_numpy(_normalized_adjacency(graph.adjacency).astype(np.float32))
        else:
            self._a = torch.from_numpy(graph.adjacency.astype(np.float32))

        params = list(self.layer1.parameters()) + list(self.layer2.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        self.train_loss = []
        best_loss, stale = float("inf"), 0
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            hidden = self.layer1(self._a, x)
            out = self.layer2(self._a, hidden)
            loss = loss_fn(out[mask], y_t[mask])
            loss.backward()
            optimizer.step()
            value = float(loss.item())
            self.train_loss.append(value)
            if value < best_loss - 1e-6:
                best_loss, stale = value, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self._graph = graph
        return self

    def predict(self, graph: StationGraph | None = None) -> np.ndarray:
        torch = _require_torch()
        graph = graph or self._graph
        x = torch.from_numpy(graph.features.astype(np.float32))
        with torch.no_grad():
            hidden = self.layer1(self._a, x)
            out = self.layer2(self._a, hidden)
        return out.numpy().reshape(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in list(self.layer1.parameters()) +
                  list(self.layer2.parameters()))


class GCNModel(_DenseGraphModel):
    """Two-layer GCN for node regression - infrastructure only, see module docstring."""

    def __init__(self, **kwargs):
        super().__init__(GCNLayer, **kwargs)


class GATModel(_DenseGraphModel):
    """Two-layer GAT for node regression - infrastructure only, see module docstring."""

    def __init__(self, **kwargs):
        super().__init__(GATLayer, **kwargs)


# ---------------------------------------------------------------------------
# Station topology - the actual blocker
# ---------------------------------------------------------------------------


class MissingTopologyError(LookupError):
    """Raised by :func:`require_station_topology` when no real hydrological
    edge list is on record for a set of stations."""


#: No hydrological edge list (which station's reach drains into which) has
#: been sourced for the Ayase or Naka monitoring stations - see
#: docs/DATA_LIMITATIONS.md Phase 8. Deliberately empty rather than
#: populated from geographic proximity, which the spec this project follows
#: explicitly rules out as a substitute for a real hydrological relationship.
STATION_EDGES: dict[str, list[tuple[str, str]]] = {}


def require_station_topology(station_names: list[str], river: str) -> list[tuple[str, str]]:
    """Return the ``(upstream, downstream)`` edge list for ``river``, or raise.

    This is the pipeline's entry guard: every layer above is usable and
    tested today, but none of them can run on a real AquaNexus station
    network until this stops raising for it.
    """
    if river not in STATION_EDGES:
        raise MissingTopologyError(
            f"no hydrological edge list on record for river {river!r} "
            f"({len(station_names)} candidate node(s): {sorted(station_names)}). "
            "See docs/DATA_LIMITATIONS.md Phase 8."
        )
    return STATION_EDGES[river]
