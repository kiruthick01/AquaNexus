# Data Limitations

This document states, for each new capability requested in `docs/ML_ROADMAP.md`,
exactly what data AquaNexus is missing and what would be needed to unblock it. It
exists so a blocked phase is a documented, falsifiable claim rather than an
assumption - see `scripts/phase1_forecasting_feasibility.py` for the script that
produced the Phase 1 numbers below by actually running the infrastructure against
the real dataset, not by inspection alone.

---

## Phase 1 — Time-series forecasting: blocked, with a measured minimum to unblock

**What exists:** 138 real dissolved-oxygen observations across 4 Ayase stations,
2022-04-07 to 2025-03-07, timestamped. Per station: 52内匠橋 n=48, 54槐戸橋 n=36,
55畷橋 n=36, 57綾瀬川合流点前 n=18 — irregular grab sampling roughly monthly, with
gaps of weeks to months.

**What was measured:** running `src/aquanexus/ml/forecasting.py`
(`add_lag_features` + `add_lead_targets`, relative tolerance ±20% of the nominal
lag/horizon, floored at 1 day — see that module's docstrings for why a relative
rather than fixed tolerance is used) against the canonical dataset:

| Horizon | Rows with lag(1,3,7d) feature *and* lead(h) target within tolerance |
|---|---|
| DO(t+1) | 0 / 138 |
| DO(t+3) | 0 / 138 |
| DO(t+7) | 0 / 138 |

Even the single-day lag alone is only satisfiable for 12/138 rows before a
matching lead target is required; requiring both a complete lag feature set and
an in-tolerance target leaves zero usable rows at every horizon. No forecasting
model can be trained or evaluated on this data honestly.

**What would unblock it, and why that number:** a walk-forward evaluation needs
multiple non-overlapping chronological folds to say anything about
generalization across time. Taking a minimally defensible protocol of 5 folds
with at least 10 test points per fold (small enough to be attainable, large
enough that a single-fold RMSE is not dominated by 1-2 points): **at least
~50-100 real observations per station, at a fixed regular sampling interval
(daily or weekly — not ad hoc monthly grabs), spanning at least 2 full seasonal
cycles.** This is derived from 5 × 10 = 50 as a floor, doubled to allow for the
train side of the split and for missing/rejected rows at the ±20% tolerance;
it is not a round number chosen for convenience. None of the three ingested
monitoring years meet this, because the underlying record is regulatory grab
sampling, not continuous sensor logging — a program design fact, not a
modelling shortfall.

**What is safe to do in the meantime:** the lag/lead/rolling/walk-forward
infrastructure itself is generic and tested (`tests/test_forecasting.py`,
synthetic series only). If a denser real series becomes available — for
example a continuous sonde deployment — Phase 1 can be attempted immediately
without rebuilding this layer.

---

## Phase 4 — Satellite / remote sensing: blocked by missing station coordinates

The observation dates (2022-2025) sit well inside Sentinel-2 (2015-) and
Landsat coverage, so temporal compatibility is not the blocker.

**What is missing:** no latitude/longitude exists anywhere in this repository
for the water-quality monitoring stations — only Japanese station names
(`52内匠橋`, `55畷橋`, `54槐戸橋`, `57綾瀬川合流点前`, plus 5 Naka stations).
`pointcloud.py` carries lon/lat helpers for bathymetry *tiles*, and
`ARCHITECTURE.md` fixes `EPSG:6677` for point-cloud geometry, but neither
attaches a coordinate to a monitoring station.

**What was measured:** running `aquanexus.remote_sensing.
require_station_coordinates` (`scripts/phase4_remote_sensing_feasibility.py`)
against the real station lists from both rivers: **9/9 stations** (4 Ayase +
5 Naka) have no coordinate on record. See `docs/REMOTE_SENSING.md` for the
implemented-but-blocked feature-extraction infrastructure (spectral indices,
cloud filtering, spatial buffers, temporal matching - all tested against
synthetic data).

**What would unblock it:** the actual station coordinates, sourced from a
citable public record (e.g. the Saitama prefecture monitoring program's own
station registry) and documented with their provenance. Estimating coordinates
from station names or approximate river-km position would introduce
unverifiable spatial error into every downstream buffer/NDVI feature and must
not be done as a substitute.

---

## Phase 8 — Graph Neural Network: blocked by network size and missing topology

**What is missing:** only 4 real Ayase monitoring stations (5 for Naka) exist
as potential graph nodes, and no upstream/downstream edge list between them
is encoded anywhere — `river_station` in the HEC-RAS data orders cross-sections
*within one reach* for hydraulic interpolation, which is a different graph
than the monitoring network.

**What was measured:** running `aquanexus.ml.graph.require_station_topology`
(`scripts/phase8_gnn_feasibility.py`) against the real station lists from
both rivers: 0 rivers have a recorded edge list. The sweep's `river_station`
field was checked directly, not assumed unusable — it carries 49 distinct
values for the Ayase and 35 for the Naka, but none is joined to a monitoring
station's identity anywhere in `aquanexus.data.dataset`, confirming it is a
different graph (cross-sections within a reach) than the one this phase
needs (the monitoring network). See `docs/GNN.md`.

**What would unblock it:** both (a) enough monitored nodes for message passing
to be meaningful (single digits is not), and (b) an actual hydrological edge
list — which station's reach drains into which — from a source other than
geographic proximity, since proximity without confirmed flow direction is not
a scientifically justified edge.

---

## Phases not listed here

Phases 2 (uncertainty), 3 (conformal prediction), 6 (transfer learning), and 7
(Bayesian modeling) are viable on the existing data; any limitations specific
to them are documented in their own `docs/UNCERTAINTY.md`,
`docs/TRANSFER_LEARNING.md`, and `docs/BAYESIAN_MODELING.md` as those phases
are implemented, not here. Phase 5 (deep learning) is partially viable — see
`docs/ML_ROADMAP.md` §4 for the MLP/LSTM split.
