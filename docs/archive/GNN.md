# Graph Neural Networks (Phase 8)

## Status: infrastructure implemented, blocked on network size and topology

Run `scripts/phase8_gnn_feasibility.py` against the real Ayase and Naka
canonical datasets:

```
PHASE 8 GNN: BLOCKED BY MISSING TOPOLOGY (both rivers)
Ayase: 4 candidate node(s), no edge list.
Naka: 5 candidate node(s), no edge list.
```

Two separate blockers, both real:

1. **Network size.** 4 real Ayase stations, 5 Naka - single digits either
   way. A GCN/GAT over that few nodes cannot produce a message-passing
   result that is evidence about river network structure rather than an
   artifact of a toy graph.
2. **Missing topology.** No upstream/downstream edge list between
   *monitoring stations* exists anywhere in this repository. The one
   topology-like field that does exist - `river_station` in the HEC-RAS
   sweep - is a different graph entirely: it orders **cross-sections** along
   one continuous reach for hydraulic interpolation (49 distinct values for
   the Ayase sweep, 35 for the Naka), and is never joined to a monitoring
   station's identity anywhere in `aquanexus.data.dataset`. Confirmed
   directly by the feasibility script, not assumed from inspection alone.

Either blocker alone would be sufficient; both apply. Connecting stations by
geographic proximity instead of a real hydrological relationship - which
station's reach drains into which - is explicitly not an acceptable
substitute (the spec this project follows requires a scientific
justification for every edge, not mere closeness), so no edge list was
fabricated to work around this.

## What is implemented (`src/aquanexus/ml/graph.py`)

All of the following are implemented and tested against a synthetic 6-node
graph (`tests/test_graph.py`, 12 tests) - none has been run against a real
station network, because none can be, per the two blockers above:

- **`StationGraph`** / **`build_station_graph`**: an explicit-edge-list graph
  representation. Edges are always supplied by the caller as named
  `(upstream, downstream)` pairs - the API has no notion of "nearby", by
  design.
- **`GCNLayer`**: dense graph convolution using Kipf & Welling's (2017)
  renormalization trick (`D^-1/2 (A+I) D^-1/2`).
- **`GATLayer`**: dense graph attention (Velickovic et al. 2018) - attention
  weights computed for every neighbor pair and masked to the actual
  adjacency (plus self-loops).
- **`GCNModel`** / **`GATModel`**: two-layer node-regression wrappers around
  the layers above, with transductive semi-supervised training (all nodes'
  features and the fixed graph are visible every forward pass; only the
  *loss* is masked to training nodes, so a held-out node's own label never
  trains the model but its neighbors' messages still reach it - verified in
  `tests/test_graph.py::test_held_out_node_never_seen_in_training_loss`).
- **The entry guard**: `require_station_topology`, which every future
  real-network call must pass through, and which is exactly what raises
  today.

Both layers are hand-rolled in plain PyTorch (already a dependency via the
`deep` extra) rather than adding `torch_geometric`: at 4-10 nodes, a dense
adjacency matrix needs nothing a sparse-graph library provides, and
`torch_geometric`'s additional compiled extensions are a real, avoidable
installation risk for a graph this small.

## What would unblock this phase

1. Enough monitored nodes for message passing to be meaningful - single
   digits is not (this is a monitoring-program-design ceiling, not something
   more modeling effort can fix).
2. An actual hydrological edge list - which station's reach drains into
   which - from a source with real flow-direction information (e.g. a river
   network dataset or the monitoring program's own documentation), not
   inferred from geographic coordinates alone even if those become
   available (Phase 4).

## What must not happen

- No GCN/GAT result should be reported for the real Ayase or Naka network
  until `require_station_topology` stops raising for it.
- No edge should be added based on geographic proximity as a substitute for
  a real hydrological relationship.
- No comparison "GNN vs. non-graph model" should be reported without both
  blockers above being resolved first - a comparison run anyway would be
  comparing a real model against an artifact of a 4-node toy graph, not
  evidence about network structure.
