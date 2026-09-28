"""Tests for the Phase 8 GNN infrastructure.

Everything here runs against a small synthetic graph - see
``src/aquanexus/ml/graph.py`` for why: no real hydrological edge list exists
for the Ayase or Naka monitoring stations, so there is no real network to
train a GCN/GAT on yet.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aquanexus.ml.graph import (  # noqa: E402
    STATION_EDGES,
    GATModel,
    GCNModel,
    MissingTopologyError,
    StationGraph,
    build_station_graph,
    require_station_topology,
)


@pytest.fixture
def chain_graph():
    """A 6-node directed chain: 0 -> 1 -> 2 -> 3 -> 4 -> 5.

    Edge (i, i+1) means adjacency[i, i+1] = 1, so row-wise aggregation
    (``A_hat @ X``) has node i's *outgoing* edge pull in node i+1's feature -
    i.e. each node aggregates its downstream neighbor. The target is built to
    need exactly that: a model with no message passing (an isolated per-node
    function of its own feature) cannot solve it, but one that aggregates a
    neighbor's feature can.
    """
    names = [str(i) for i in range(6)]
    edges = [(str(i), str(i + 1)) for i in range(5)]
    rng = np.random.default_rng(0)
    own_feature = rng.uniform(0, 1, 6).astype(np.float32).reshape(-1, 1)
    graph = build_station_graph(names, edges, own_feature, directed=True)
    # target[i] = 3 * own_feature[i+1] for i<5 (needs the downstream
    # neighbor node i+1 actually reaches via the edge above); node 5 has no
    # outgoing edge and so no neighbor to aggregate, and is excluded from
    # the training loss.
    target = np.zeros(6, dtype=np.float32)
    target[:-1] = 3.0 * own_feature[1:, 0]
    train_mask = np.array([True, True, True, True, True, False])
    return graph, target, train_mask


# ---------------------------------------------------------------------------
# StationGraph construction
# ---------------------------------------------------------------------------


def test_build_station_graph_directed_edges():
    names = ["A", "B", "C"]
    features = np.zeros((3, 1), dtype=np.float32)
    graph = build_station_graph(names, [("A", "B")], features, directed=True)
    assert graph.adjacency[0, 1] == 1.0
    assert graph.adjacency[1, 0] == 0.0


def test_build_station_graph_undirected_mirrors_edges():
    names = ["A", "B", "C"]
    features = np.zeros((3, 1), dtype=np.float32)
    graph = build_station_graph(names, [("A", "B")], features, directed=False)
    assert graph.adjacency[0, 1] == 1.0
    assert graph.adjacency[1, 0] == 1.0


def test_build_station_graph_rejects_unknown_node():
    names = ["A", "B"]
    features = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown node"):
        build_station_graph(names, [("A", "Z")], features)


def test_station_graph_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        StationGraph(node_names=["A", "B"], adjacency=np.zeros((2, 2)),
                    features=np.zeros((3, 1)))


# ---------------------------------------------------------------------------
# GCN / GAT models
# ---------------------------------------------------------------------------


def test_gcn_fit_predict_shape(chain_graph):
    graph, target, train_mask = chain_graph
    model = GCNModel(hidden_size=8, max_epochs=200, patience=50, seed=0).fit(
        graph, target, train_mask
    )
    predictions = model.predict()
    assert predictions.shape == (6,)


def test_gat_fit_predict_shape(chain_graph):
    graph, target, train_mask = chain_graph
    model = GATModel(hidden_size=8, max_epochs=200, patience=50, seed=0).fit(
        graph, target, train_mask
    )
    predictions = model.predict()
    assert predictions.shape == (6,)


def test_gcn_uses_message_passing_not_just_own_feature(chain_graph):
    """The target needs a neighbor's feature; training loss should drop well
    below the variance of an own-feature-only (no-message-passing) guess."""
    graph, target, train_mask = chain_graph
    model = GCNModel(hidden_size=8, max_epochs=300, patience=100, seed=1).fit(
        graph, target, train_mask
    )
    final_loss = model.train_loss[-1]
    naive_variance = float(np.var(target[train_mask]))
    assert final_loss < 0.5 * naive_variance


def test_gcn_reports_parameter_count(chain_graph):
    graph, target, train_mask = chain_graph
    model = GCNModel(hidden_size=4, max_epochs=10, seed=0).fit(graph, target, train_mask)
    assert model.n_parameters() > 0


def test_held_out_node_never_seen_in_training_loss(chain_graph):
    graph, target, train_mask = chain_graph
    held_out = ~train_mask
    model = GCNModel(hidden_size=8, max_epochs=200, seed=0).fit(graph, target, train_mask)
    # Node 5 (held out) has no downstream neighbor and an unconstrained
    # target of 0 in this fixture; the model must still produce a finite
    # prediction for it via the forward pass even though its loss never
    # trained on it.
    predictions = model.predict()
    assert np.all(np.isfinite(predictions[held_out]))


# ---------------------------------------------------------------------------
# Station topology - the actual Phase 8 blocker
# ---------------------------------------------------------------------------


def test_station_edges_registry_is_currently_empty():
    assert STATION_EDGES == {}


def test_require_station_topology_raises_for_ayase():
    with pytest.raises(MissingTopologyError, match="Ayase"):
        require_station_topology(["52内匠橋", "54槐戸橋", "55畷橋", "57綾瀬川合流点前"],
                                 river="Ayase")


def test_require_station_topology_succeeds_once_populated(monkeypatch):
    monkeypatch.setitem(STATION_EDGES, "test-river", [("A", "B")])
    edges = require_station_topology(["A", "B"], river="test-river")
    assert edges == [("A", "B")]
