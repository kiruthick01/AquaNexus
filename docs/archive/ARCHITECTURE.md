# Architecture

## Data flow

```
Saitama river point cloud (LAS, multibeam + UAV)   MLIT gauges      JMA
                │                                       │            │
                ▼                                       ▼            ▼
        channel geometry                        discharge BCs   met forcing
                │                                       │            │
                └───────────────┬───────────────────────┴────────────┘
                                ▼
                    HEC-RAS 6.x  (hydraulics + temperature)
                                │
                                ▼
                   simulated environmental states
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   MOE monthly observations              synthetic HSI labelling
   (bounds + realism check)              (ecological response curves)
              └─────────────────┬─────────────────┘
                                ▼
                       feature engineering
                                │
                                ▼
                   XGBoost regressor  +  SHAP
                                │
                                ▼
                     FastAPI  →  React dashboard
```

## Design decisions

### Designed experiment, not a time series
The original plan called for 35,040 hourly records across a continuous 2020–2023
simulation. That is not supportable: water-quality observations are monthly, so hourly
boundary conditions for nitrogen, phosphorus, CBOD and algae would be interpolation
artefacts, and the model would learn the interpolation scheme rather than the river.

Instead, HEC-RAS sweeps a **designed grid over flow space** — discharge × temperature ×
water-quality conditions, bounded by observed ranges. Every row is a physically consistent
simulated state, and the dataset makes no claim to be a chronology. This trades a false
temporal richness for defensible coverage of the state space.

Consequence: the temporal validation split from the original plan does not apply. Validation
is by **held-out region of flow space** and **held-out reach** instead. See
`docs/ML_METHODOLOGY.md`.

### Hydraulics simulated, water quality observed
HEC-RAS is run for hydraulics and heat transport, where boundary conditions genuinely exist
at usable resolution. The NSMI nutrient module is available in HEC-RAS 6.x but requires
per-constituent boundary time series that the monthly monitoring record cannot supply.
Nutrient and DO conditions are therefore sampled from observed distributions rather than
routed, and that limitation is stated wherever results are reported.

### Configuration
A single `pydantic-settings` object (`aquanexus.config.settings`) reads `.env` and
environment variables. Nothing else reads the environment. `ensure_dirs()` is explicit
rather than import-time so that importing config never touches the filesystem — this keeps
tests hermetic.

### Coordinate reference system
`EPSG:6677` (JGD2011 / Japan Plane Rectangular CS Zone 9) throughout, matching the Saitama
point-cloud release. Reprojection happens only at ingest.

## Module boundaries

| Package | Responsibility | Depends on |
|---|---|---|
| `config`, `logger` | settings, logging | — |
| `hecras` | read/parse/run/validate HEC-RAS projects | config, logger |
| `data` | load, preprocess, validate, synthesise labels | config, logger |
| `ml` | train, explain, evaluate, predict | data, config |
| `api` | HTTP surface | ml, config |
| `dashboard` | Streamlit demo | api or ml |

Dependencies point one direction only: `api → ml → data → config`. No module imports from
a layer above it.
