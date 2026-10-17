# Results

Every number in the README comes from a CSV in `experiments/`. This file says which and how
each was produced.

## Evaluation regimes

All defined in `aquanexus.ml.do_experiment`.

| Regime | Train | Test | Question |
|---|---|---|---|
| `temporal` | FY2013-19, all stations | FY2021-22, same stations | Does it forecast ahead at monitored sites? |
| `station` | 80% of stations, all years | 20% unseen stations | Does it work at a new site? |
| `strict` | 80% of stations, FY≤2019 | unseen stations, FY2021-22 | Both at once. **The headline.** |
| `prefecture` | 4 of 5 prefecture groups | the held-out group | Does it travel to a new region? |

FY2020 is purged from `temporal` and `strict` so a test year is never adjacent to training.
Early stopping uses stations carved out of the training set, never test rows.

## Feature sets

`env` (season, coordinates, weather), `geo` (`env` + elevation and relief), `spatial`
(`geo` + other stations' earlier mean DO within 25 km), `nowcast_basic` (`spatial` + their readings
within 10 days), `nowcast` (`nowcast_basic` + the same summaries over stations within 5 km and over
stations on a same-named river within 60 km), `history` (`geo` + the station's own earlier
readings), `full_basic` (`nowcast_basic` + `history`), `full` (`nowcast` + `history`), `concurrent`
(`geo` + same-sample pH, BOD, SS; a soft sensor, not a forecast). The served `nowcast` and `full`
models use the same-named sets.
Neighbour and history features use only other visits, never the sample's own label; a test in
`tests/test_features.py` checks that tampering with a station's labels leaves its own neighbour
features unchanged.

## Files

| File | Contents |
|---|---|
| `do_results_default.csv` | All baselines and default LightGBM, all feature sets and regimes |
| `do_results_tuned_<regime>.csv` | Optuna-tuned LightGBM (30 trials, station-grouped CV) for `nowcast` and `full` |
| `do_results_tail_<regime>.csv` | Low-DO sample weights 3, 10 and 30 |
| `do_uncertainty.csv`, `do_uncertainty_groups.csv` | Conformal interval coverage overall and by group |
| `do_tail_ranking.csv` | Low-DO classifier against thresholded regression: AUC, average precision, recall at 50% precision, Brier, calibration error; with and without isotonic recalibration |
| `do_tail_intervals.csv` | Constant, binned and quantile-conformal (CQR) interval coverage, overall and for DO below 5 |
| `do_neighbor_leak_check.csv` | Re-scores with held-out stations removed from every neighbour pool |
| `do_shap_<set>.csv` | Mean absolute SHAP per feature |
| `do_breakdown.csv` | Strict-regime scores by region, season and DO band |
| `do_case_studies.csv` | Ayase and Naka stations held out entirely |
| `results.csv` | The earlier two-river study; kept for the archived documents |

## Reading them honestly

- R² is dominated by the many ordinary samples. Always read `mae_low`, `recall_low` and
  `precision_low` (DO below 5 mg/L; an alarm is a prediction below 6) beside it.
- Tuning gains are within noise on unseen stations. Tail weighting trades precision for recall.
- Interval coverage is marginal. It is near 90% overall and much lower for DO below 5 mg/L.
  Binning the width by predicted DO (served) and quantile-conformal intervals both help only a little.
- The below-5 probability is the raw classifier output. Isotonic recalibration made calibration error
  worse (0.001-0.004 raw, 0.005-0.015 recalibrated), so it is not applied.
- The served bundles train on all years, so their stored accuracy is the strict-regime result
  of the same recipe, not a score on their own training data.
