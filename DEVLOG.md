# AquaNexus development log

The log for the nationwide dissolved-oxygen rebuild. The log of the earlier two-river
version (September 2026) is in `docs/archive/DEVLOG_legacy.md`; that code is at the git tag
`legacy-ayase-naka`.

## 2026-10-16 to 2026-10-18: rebuilding around a nationwide dataset

### Why

An honest review of the first version found that its real signal was 138 dissolved-oxygen
labels from two rivers; everything else was infrastructure, a synthetic habitat index
(circular by construction) and a hydraulic model that did not move the score. The rebuild
keeps real labels only and trains on the Ministry of the Environment's nationwide record.

### Data (details in `docs/DATA_SOURCES.md`)

- 565,674 DO samples, 5,883 stations, 47 prefectures, FY2013-22. Gate asked for about 50,000.
- Weather joined from NASA POWER (193 grid cells). Open-Meteo ERA5 was tried first and abandoned
  because its free tier rate-limits long histories; ERA5-Land's daily endpoint returns only
  temperature.
- Station elevation from GSI; relief among stations within 10 km derived from it. Open-Meteo's
  elevation API was also rate-limited and dropped.
- File quirks worth remembering: `NENDO` is fiscal year and `FISCALYEAR` is calendar year of the
  sample; `DO_3` is label-derived and must never be a feature.

### Method

Four evaluation regimes (temporal, unseen station, both at once, unseen prefecture), baselines
(climatology, persistence, ridge), LightGBM with early stopping on carved-out stations,
Optuna tuning scored by station-grouped CV on training rows only, split-conformal intervals.
Seven feature sets from weather-only to weather plus neighbours plus the station's own history.

### Results

See `README.md` and `docs/RESULTS.md`. In short: strict-regime R² 0.48 (`geo`), 0.51 (`nowcast`),
0.72 (`full`) against 0.38 for the best baseline.

What did and did not work:

- Station identity is worth about 0.25 R². Weather alone loses to climatology at stations the
  climatology has seen, and beats it at stations it has not.
- Elevation and relief added about 0.06-0.07 R² for weather-only models.
- Neighbour-station features added little (at most 0.05 R²) and hurt on the Ayase.
- Tuning moved held-out R² by less than 0.01 on unseen stations. Defaults are served.
- Tail weighting (weights 3, 10, 30) raises recall of DO below 5 mg/L at a steep cost in
  precision; weight 10 is served as `full_alert`.
- Interval coverage: about 0.88 at 0.90 nominal overall, 0.45-0.56 for samples below 5 mg/L.

### Integrity check: neighbour features and held-out stations

`add_neighbors` initially used every station's readings, so a training row's neighbour features
could contain a held-out station's label. That does not reach a test row's own prediction, but it
is test labels inside training inputs. `scripts/check_neighbor_leak.py` rebuilds the frame with the
held-out stations excluded from every neighbour pool. Result
(`experiments/do_neighbor_leak_check.csv`): R² moved by at most 0.011 in either direction (strict
`nowcast` 0.507 to 0.518, `full` 0.715 to 0.717, `geo` 0.484 to 0.482), so the published numbers
stand. The pooled version is kept as the default because it reflects deployment, where the
neighbours of a new station are the stations that exist.

### Cleanup and rebuild of the repository

- Removed HEC-RAS, point-cloud, GNN, Bayesian, deep-learning, transfer, remote-sensing, fish
  survey and synthetic-data code, with their tests and scripts. Old documents moved to
  `docs/archive/`. Tag `legacy-ayase-naka` preserves the old state.
- New API (`/predict`, `/models`, `/stations`) and frontend (overview, predict, models, about).
- Freed about 80 GB by deleting regenerable extracted LiDAR files and caches (with confirmation).

### Open items

- Land use / catchment data and distance to coast would need further downloads.
- Low-oxygen events remain poorly predicted; a classifier and adaptive intervals are the next
  things to try.
