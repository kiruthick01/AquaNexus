# AquaNexus: predicting dissolved oxygen in Japanese rivers

## Question

How well can dissolved oxygen (DO) in a Japanese river be predicted at a place and time the model
has never seen, using only data that is openly available nationwide?

## Data

- **Labels:** 565,674 DO measurements from 5,883 river stations in all 47 prefectures, fiscal
  years 2013-22, from the Ministry of the Environment's public water body survey. Real
  measurements only; nothing is synthetic.
- **Inputs:** NASA POWER daily weather, station elevation and local relief (GSI), land-use shares
  within 1 and 5 km (国土数値情報, FY2016), and readings from nearby stations and from the
  station's own earlier visits where a model is allowed to use them.
- **Not available:** water temperature and river flow are not in the monitoring files, so neither
  is an input.

## Method

LightGBM regression, with a separate classifier for DO below 5 mg/L and split-conformal
prediction intervals. Four honest tests keep test stations and test years out of training. The
headline is the strict one: 20% of stations are held out whole, and the test years (FY2021-22)
come after the training years. Neighbour and history features use only other visits, and a
check that removes held-out stations from every neighbour pool moved R² by at most 0.011.

## Results (unseen stations, later years)

| Model | Needs | R² | MAE (mg/L) | AUC for DO < 5 |
|---|---|---|---|---|
| Station-month climatology | station history | 0.38 | 1.17 | |
| `geo` | date, place, weather | 0.50 | 1.00 | 0.905 |
| `nowcast` | plus a nearby oxygen reading | 0.56 | 0.92 | 0.921 |
| `full` | plus the station's own history | 0.73 | 0.72 | 0.968 |

Reading these:

- A new site with no history gets about half the variance from weather, terrain and land use
  alone, and beats a climatology that has seen the site's history.
- The station's own history is the largest single gain (about 0.25 R² over `geo`).
- Elevation and relief added 0.06-0.07 R². Land use added 0.02 for `geo` and 0.01 for `nowcast`,
  and nothing once station history is known. Hyperparameter tuning moved R² by less than 0.01.
- Two rivers held out whole (Ayase, Naka) score 0.60 and 0.43 with `geo`, rising to 0.76 and 0.65
  with `full`.

## Limits

- **Low oxygen is the weak spot.** Only 1.7% of samples are below 5 mg/L. At best the models
  catch 23% (`geo`) to 57% (`full`) of them at 29-43% precision. A heavier-weighted alert model
  catches 74% at 27% precision.
- **Intervals under-cover the tail.** They reach about 89% overall at 90% nominal, but only
  40-57% for samples below 5 mg/L.
- Samples are monthly: this predicts oxygen on sampling days, not continuously.
- Land use is FY2016 in circular buffers, not true upstream catchments.
- Proof of concept. Not regulatory or safety advice.

## What would help next

Upstream catchments (needs flow-direction data), distance to the coast, and above all water
temperature or flow records. Without those, the remaining gains from more features are small.

## Reproducing

Every figure traces to a CSV in `experiments/`; `docs/RESULTS.md` says which, and
`docs/DATA_SOURCES.md` says how each source was obtained. Code and the full development log
are in the repository.
