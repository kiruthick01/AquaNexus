# AquaNexus

**Predicting dissolved oxygen in Japanese rivers from about 565,000 real monitoring
samples, and measuring honestly how far the predictions can be trusted.**

Gradient-boosted models, trained on the Ministry of the Environment's public water body
survey (5,883 river stations, all 47 prefectures, fiscal years 2013-22), joined to daily
weather and station elevation. Every accuracy figure comes from stations the model never
saw, in years after the ones it trained on. A FastAPI service serves the models with
90% prediction intervals and exact per-feature contributions; a React dashboard sits on top.

> This replaces an earlier version that trained on 138 samples from two Saitama rivers and
> a synthetic habitat index. That version is preserved at the git tag `legacy-ayase-naka`
> and its documents are in `docs/archive/`.

## Results

Scored on stations **unseen in training**, in fiscal years 2021-22 (training used years up to
2019; 2020 is purged). 22,130 test samples. Reproduce with `scripts/train_do.py`.

| Model | What it needs | R² | Mean error | Samples < 5 mg/L caught | Alarms that are right |
|---|---|---|---|---|---|
| climatology (baseline) | station history | 0.38 | 1.17 mg/L | n/a | n/a |
| persistence (baseline) | station history | 0.36 | 1.19 mg/L | n/a | n/a |
| `geo` | coordinates, date, weather | 0.48 | 1.02 mg/L | 20% | 37% |
| `nowcast` | plus neighbouring stations' recent readings | 0.51 | 0.99 mg/L | 19% | 36% |
| `full` | plus the station's own earlier readings | **0.72** | **0.75 mg/L** | 57% | 43% |
| `full_alert` | `full`, weighted toward low oxygen | 0.69 | 0.79 mg/L | 77% | 26% |

What the numbers say, without rounding in our favour:

- The models beat naive baselines on stations they have never seen. Station history is the
  largest single gain (about +0.2 R²), but it needs earlier readings from the same station.
- A brand-new station with only weather and terrain explains about half the variance.
- **Low oxygen is the hard part.** Only 2.2% of samples are below 5 mg/L. Without tail
  weighting the models catch under a fifth of them (for `geo`) and with it they trade
  precision for recall. Conformal intervals cover about 88% of cases overall but only
  45-56% of samples below 5 mg/L.
- Hyperparameter tuning (Optuna, 30 trials, station-grouped CV) moved held-out R² by less than
  0.01 on unseen stations, so the served models use defaults.
- Accuracy by region, season and DO band is in `experiments/do_breakdown.csv`. Summer is the
  hardest season (R² 0.52 for `full`) and Chugoku and Kyushu the hardest regions (0.61).
- On the Ayase (綾瀬川) and Naka (中川), held out entirely, `full` scores R² 0.73 and 0.65;
  `geo` scores 0.51 and 0.34 (`experiments/do_case_studies.csv`; small samples, so noisy).

All result tables are CSVs in `experiments/`; `docs/RESULTS.md` explains each.

## Data

| Layer | Source | Notes |
|---|---|---|
| Dissolved oxygen | 環境省 水環境総合情報サイト, 公共用水域水質測定 (river) | 565,674 samples, FY2013-22. Downloaded by hand through the site's web form. |
| Weather | NASA POWER (MERRA-2, GPM IMERG) | Daily air temperature, rain, solar, wind on a 0.5° grid, fetched by `scripts/fetch_weather.py`. |
| Elevation | 国土地理院 elevation API | One value per station, `scripts/fetch_elevation.py`. |

The Ministry's site prohibits automated acquisition, so the monitoring files are downloaded
manually and `river.go.jp` is never scraped. See `docs/DATA_SOURCES.md`.

## Run it

```bash
pip install -e ".[ml,api,dev]"
pytest                                   # no data needed; tests build toy models
```

To reproduce the models you need the monitoring CSVs in `data/raw/moe/` (see
`docs/DATA_SOURCES.md`), then:

```bash
python scripts/fetch_weather.py          # resumable; caches to data/cache/
python scripts/fetch_elevation.py
python scripts/train_do.py --regimes strict --sets geo nowcast full
python scripts/save_do_model.py          # bundles in data/models/do/
uvicorn aquanexus.api.app:app --reload   # http://localhost:8000/docs
cd frontend && npm install && npm run dev
```

## Layout

```
src/aquanexus/data/   moe.py (loader), weather.py, geography.py (terrain), features.py
src/aquanexus/ml/     do_experiment.py (splits, baselines, LightGBM, tuning, conformal), do_bundle.py
src/aquanexus/api/    FastAPI app: /predict, /models, /stations
scripts/              fetching, training, evaluation, saving; see each docstring
experiments/          every reported number as a CSV
frontend/             React dashboard
```

## Limits

Monthly samples mean predictions are for sampling days, not continuous. There is no water
temperature or flow in the monitoring files. Intervals under-cover in the low-oxygen tail.
This is a proof of concept, not regulatory or safety advice.

## Licence and attribution

MIT for the code. Water quality data: 環境省 (Ministry of the Environment, Japan). Weather: NASA
POWER project (MERRA-2, GPM IMERG). Elevation: 国土地理院 (Geospatial Information Authority of Japan).
