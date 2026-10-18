# AquaNexus

**Predicting dissolved oxygen in Japanese rivers from about 565,000 real monitoring
samples, and measuring honestly how far the predictions can be trusted.**

Gradient-boosted models, trained on the Ministry of the Environment's public water body
survey (5,883 river stations, all 47 prefectures, fiscal years 2013-22), joined to daily
weather and station elevation. Every accuracy figure comes from stations the model never
saw, in years after the ones it trained on. A FastAPI service serves the models with
90% prediction intervals and exact per-feature contributions; a React dashboard sits on top.
A one-page summary is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

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
| `geo` | coordinates, date, weather; terrain and land use come from the location | 0.50 | 1.00 mg/L | 23% | 29% |
| `nowcast` | plus neighbouring stations and same-river stations' recent readings | 0.56 | 0.92 mg/L | 32% | 34% |
| `full` | plus the station's own earlier readings | **0.73** | **0.72 mg/L** | 57% | 43% |
| `full_alert` | `full`, weighted toward low oxygen | 0.70 | 0.77 mg/L | 74% | 27% |

Each model also returns a **probability that DO is below 5 mg/L** from a dedicated classifier
(AUC 0.91 / 0.92 / 0.97 for `geo` / `nowcast` / `full`, against 0.86 / 0.88 / 0.95 from thresholding
the regression; the base rate is 1.7%, and the probabilities are well calibrated, expected
calibration error 0.001-0.004).

What the numbers say, without rounding in our favour:

- The models beat naive baselines on stations they have never seen. Station history is the
  largest single gain (about +0.2 R²), but it needs earlier readings from the same station.
- A brand-new station with only weather and terrain explains about half the variance.
  Land use within 1 and 5 km of the station added about +0.02 R² (unseen stations; 0.484 to 0.504)
  and what neighbouring and same-river stations read recently lifts it to 0.56. Land use did not
  help models that already have the station's own history.
- **Low oxygen is the hard part.** Only 1.7% of strict-test samples are below 5 mg/L.
  Without tail weighting `geo` catches a fifth of them. Weighting trades precision for
  recall. Intervals cover about 89% of cases overall but only 40-57% of samples below
  5 mg/L (`geo` to `full`); conformalized quantile regression barely helps (up to 58%).
- Hyperparameter tuning (Optuna, 30 trials, station-grouped CV) moved held-out R² by 0.004 or less
  on unseen stations, so the served models use defaults.
- Summer is the hardest season (`full` R² 0.54) and Kyushu-Okinawa and Chugoku the hardest regions
  (0.63, 0.64). Every region and season beats climatology. See `experiments/do_breakdown.csv`.
- On the Ayase (綾瀬川) and Naka (中川), held out entirely: `geo` R² 0.60 and 0.43, `nowcast` 0.66
  and 0.44, `full` 0.76 and 0.65 (small samples, so noisy; `experiments/do_case_studies.csv`).
- Neighbour features were checked for leakage from held-out stations' labels: excluding them
  changed R² by at most 0.011 (`experiments/do_neighbor_leak_check.csv`).

All result tables are CSVs in `experiments/`; `docs/RESULTS.md` explains each.

## Data

| Layer | Source | Notes |
|---|---|---|
| Dissolved oxygen | 環境省 水環境総合情報サイト, 公共用水域水質測定 (river) | 565,674 samples, FY2013-22. Downloaded by hand through the site's web form. |
| Weather | NASA POWER (MERRA-2, GPM IMERG) | Daily air temperature, rain, solar, wind on a 0.5° grid, fetched by `scripts/fetch_weather.py`. |
| Elevation | 国土地理院 elevation API | One value per station, `scripts/fetch_elevation.py`. |
| Land use | 国土数値情報 土地利用細分メッシュ (L03-b, FY2016, 100 m) | Share of each class within 1 and 5 km of the station; `scripts/fetch_landuse.py`, `scripts/build_landuse.py`. |

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
python scripts/fetch_landuse.py          # ~920 MB from nlftp.mlit.go.jp, resumable
python scripts/build_landuse.py
python scripts/train_do.py --regimes strict --sets geo_lu nowcast_lu full
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
Land use: 出典 国土交通省 国土数値情報（土地利用細分メッシュ）を加工して作成.
