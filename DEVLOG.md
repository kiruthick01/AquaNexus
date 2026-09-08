# AquaNexus Development Log

**Project**: AquaNexus — Physics-Informed ML Framework for Aquatic Ecosystem Diagnosis  
**Developer**: kiruthick01  
**Timeline**: 2026-09-05 → 2026-09-07  
**Status**: 🟢 Complete — one claim outstanding: neither container image has been
built (no Docker daemon on this machine; CI builds both on its first run)  
**Repository**: https://github.com/kiruthick01/aquanexus  

---

---

## Daily Log

Short notes. Detail lives in `docs/` — this is just what happened when.

### 2026-09-05
- Read plan + ML strategy. No code yet.
- Flagged two risks: synthetic labels are circular; data sources unverified.

### 2026-09-06
- **Data audit.** Yodo has no open bathymetry → switched to Ayase (Saitama). river.go.jp blocks scraping. Dataset design changed to designed-experiment.
- **Scaffold + Phase 1a.** Package, config, Docker, CI-ready. Repo pushed.
- **Point cloud ingest.** Tile index recovered from vector tiles (not published as a file). 89 Ayase tiles, contiguous, 29.4 km. CRS axis order verified (x=easting; swapped is 16 km off).
- **Water quality loader.** 3,892 samples FY2022–24. ~9,400 censored values/yr handled explicitly.
- **Features + synthetic labels.** Found 4 structural errors in ML_STRATEGY §3–4 (unbounded oxygen formula, sigmoid that ignores cold, mis-ordered penalty branches, arithmetic mean masking lethal conditions). Falsification test passes on real data.
- **HEC-RAS working.** v7.0 via COM. Bank stations must exist in the station list — that was the blocker. Plan file needs ~200 keys, so templated from a reference project the user built.
- **Full reach.** 89 tiles (9.5 GB), 53 sections, 27.5 km, 3 profiles computed. Bed falls 0.39 m/km — right sign. Validator: 0 errors, 20 warnings (4 constrictions).
- **Validator, notebooks, HEC-RAS guide.** Parallel downloads (16 → 500 MB/min).
- **Phase 2a.** State vectors (7,314 rows). R² 0.99 on synthetic labels — flagged as function recovery, not skill.
- **Course correction.** Raised that the ML was circular. Found real biology data (河川水辺の国勢調査) — n=6 for Ayase, too small to train, good for validation.
- **Real ML target added.** Observed DO, n=138. R² 0.44 grouped CV. Ayase runs ~2.4 mg/L below saturation.
- **Phase 2b.** SHAP. ~~temp × discharge synergistic at −1.02 mg/L~~ (retracted 09-07, see below). 12 collinear pairs → 2 of 4 interactions unidentifiable, reported as such.
- **Phase 2c.** Validation + baselines. Model beats persistence by only 0.06 R². Under-predicts DO by 2 mg/L at low flow.
- **Phase 3a.** FastAPI up. Serves both models with provenance + caveats in every response. 25 API tests.
  - Scenario endpoint shows the low-flow flaw in action: −60% discharge *raises* predicted DO, which is physically wrong. Caveat is real, not boilerplate.
- **README + repo metadata.** 7 figures generated from real project output (`scripts/make_figures.py`), 3 mermaid diagrams. GitHub description + 18 topics set. Japanese data provenance surfaced throughout.

### 2026-09-07
- **Phase 2 notebooks.** 03 training, 04 explainability, 05 validation. All three run end to end from a clean kernel; committed unexecuted, as 01 and 02 are.
- **Retraction found while rebuilding 2b.** The −1.02 mg/L temp × discharge synergy was one station's 48 rows. Pooled over 138 it is +0.23 and the per-station sign flips (−0.93 to +1.72). Corrected in README, ML_METHODOLOGY, and here. Nothing depended on it — the API does not serve interactions.
- **Scenario diagnosis sharpened.** "Physically wrong" was too strong: DO and discharge are negatively associated *within every station* (−0.14 to −0.60), so the model reproduces the record rather than inventing a sign. What is genuinely broken is that `/scenario_run` holds depth/velocity/width fixed while discharge moves — an impossible state — and that low flow is outside usable support (bias −2 mg/L, and "low flow" is largely one shallow station).
- **Ablation reproduced exactly** against the documented ladder, plus one honest addition: adding air temperature (+0.073 R²) buys more than the whole HEC-RAS pipeline does (+0.027).
- **Phase 3b.** 35 new tests (21 integration, 14 deployment invariants) and `scripts/verify_deployment.py`. Verification turned up three real defects, all fixed:
  - `/explain` returned **0.0 for every feature** — the SHAP background was the request row itself, so baseline == prediction. Training background now ships beside each model; explainer built once (warm explain 2.4 s → 190 ms).
  - `collinear_pairs` was in the schema and never populated — the route read a nonexistent attribute. Computed at training time now.
  - `DATA_DIR` resolved into site-packages for an installed package, so the container would have ignored its `./data` mount and started degraded. Pinned in the Dockerfile; the override now propagates to the subdirectories.
  - Also: compose mounted `./src` over a non-editable install, making the mount inert. `PYTHONPATH=/app/src`.
- **Docker still unbuilt** (no daemon here). Verified by proxy: clean-venv `pip install ".[ml,api]"`, then served from the installed copy with no source on the path — degraded without `DATA_DIR`, 14/14 smoke checks with it.
- **API_REFERENCE.md written** — was a placeholder since Phase 3.
- **Phase 4.** React 19 + TypeScript dashboard (Vite), 5 pages, 7 components, 22 tests. Hand-written CSS sharing the figures' palette; inline SVG for the SHAP bars and the response surface. Verified in a real Chrome against the live API, not only in jsdom.
  - The browser found three things jsdom could not: CORS rejected the `127.0.0.1` spelling of the dev origin; the default form state opened *outside* the training range; and the UI flagged a depth the API did not, because `check_ranges` ignored the `depth → reach_depth` alias that `build_features` applies. All three fixed.
  - `/models` now serves `training_ranges`, so the frontend shows where the evidence ends instead of hard-coding numbers that would drift on retraining.
  - Frontend image (node → nginx, SPA fallback), compose service, and a CI workflow whose `containers` job is the first place either image is actually built.
  - Added the MIT LICENSE file — pyproject had declared MIT since day one with no licence text in the repo.


### 2026-09-08
- **Scenario hydraulics fixed.** `/scenario_run` re-interpolates depth, velocity and width from the sweep at the scenario discharge. Also found `reach_top_width` had been NaN on every API request since Phase 3a — no caller field maps to it — so the linear pipeline was imputing the training median, the same width at 1 m³/s as at 70.
- **`/explain` cached and capped.** A repeated state answers in ~15 ms instead of ~190; a new one spends a token from a per-client bucket. Cache hits cost nothing, because the limit protects CPU.
- **Frontend API address moved to runtime.** The entrypoint writes `/config.js` from `$API_BASE_URL`; one built image now serves any environment. Verified by repointing a built bundle in the browser.
- **Manning's n quantified.** Against the *flawed* geometry it read 1.10 mg/L, 64% of the model's error; re-run after the geometry correction below it is 0.26 mg/L, **15%**. The first number was mostly the bad sections talking. Uncalibrated is now at least ranked, and ranked below the geometry itself.
- **Four bad cross-sections measured, then removed — and the model got worse.** RS 12500/14000/18000/24000 were cut through bank where the centreline wandered. Excluding them moves predictions 1.10 mg/L at worst, so they came out and everything was retrained on 49 sections:

| | 53 sections (flawed) | 49 sections (shipped) |
|---|---|---|
| RMSE | 1.713 | **1.785** |
| R² | 0.442 | **0.394** |
| margin over persistence | +0.057 R² | **+0.009 R²** |
| low-flow bias | −2.011 | **−2.256 mg/L** |
| hydraulic gain over raw discharge | +0.062 R² | **+0.019 R²** |

  Every headline in the project got weaker, including the central claim about the physics-informed step, which the flawed geometry had inflated threefold. The numbers above replace the Phase 2 ones throughout README, ML_METHODOLOGY, the API caveats, the figures and the dashboard; the Phase 2 entries in this log are left as written, because they record what was true when they were written.
- **`scripts/run_hecras.py` implemented** — it had been a `NotImplementedError` placeholder since Phase 1. It runs the 12-discharge sweep and writes the CSV the models are built from, with `--exclude-flagged` for the above.

### Next session — pick up here

**State:** All four phases done, plus a round of open-item work. 376 backend tests + 22 frontend, lint clean, all pushed. The models were retrained on corrected geometry (49 sections) on 09-08 — R² 0.394, RMSE 1.785.

**Next, in order of value:**
1. Watch the first CI run: the `containers` job builds both images for the first time. Expect it to be where the remaining Docker unknowns surface.
2. Cheap and worth it: re-derive reach hydraulics from the sweep inside `/scenario_run` (see below).
3. Optional polish: dark mode; a shareable permalink for a state; caching `/explain` by state, since it is ~200 ms of SHAP per call.

**Known debt:**
- ~~4 cross-sections cut through constrictions~~ — **excluded 09-08** after measuring their effect (1.10 mg/L, 64% of RMSE). The sweep, both models and every documented number are now on 49 sections.
- Manning's n uncalibrated — **quantified 09-08** (`docs/MANNING_SENSITIVITY.md`): across 0.025–0.050 the predictions move up to 0.26 mg/L, **15% of the model's 1.785 RMSE**. Still uncalibrated, but ranked: a gauged rating curve remains the most valuable missing measurement, and this says it is worth less than the geometry fix already made.
- ~~`/scenario_run` holds depth/velocity/width fixed when discharge changes~~ — **fixed 09-08**: the hydraulics are re-interpolated from `ayase_flow_sweep.csv` at the scenario discharge, and `reach_top_width`, which no caller field could ever supply, no longer falls back to an imputed median on every request.
- Scenario answers below ~2 m³/s should not be believed regardless — the model is biased −2.26 mg/L there and drought mechanisms (heat, residence time, concentrated load) are not in the feature set.
- Model beats persistence by **0.009 R²** and loses on MAE; predicted range 4.2–10.2 mg/L against an observed 3.0–17.0, so it cannot flag hypoxic events. Both got worse when the geometry was corrected.
- HSI labels remain synthetic; only the falsification test constrains them.
- Neither image has been built. `tests/test_deployment.py` (20 tests) covers what a daemon is not needed for; the base images, Linux wheels, libgomp, the non-root user against a bind mount, nginx's reading of its own config and the HEALTHCHECK loops are unverified. CI's `containers` job is where that gets settled.
- ~~The frontend bakes its API URL in at build time~~ — **fixed 09-08**: the container entrypoint writes `/config.js` from `$API_BASE_URL` and the page reads it at load, verified by repointing a built bundle in the browser without rebuilding.
- No auth and no TLS. `/explain` is now cached by state (~15 ms on a repeat) and capped per client, but the limit is per-process, so a shared one needs a gateway. See `docs/DEPLOYMENT.md`.
- The full debt register is also in `05_model_validation.ipynb` §7, so it travels with the analysis.

**To rebuild anything:**
```
python scripts/download_data.py                     # water quality
python scripts/build_geometry.py --river ayasegawa  # tiles -> HEC-RAS -> run
python scripts/train_models.py                      # both models + manifest
python scripts/make_figures.py                      # README figures
uvicorn aquanexus.api.app:app --reload
python scripts/verify_deployment.py                 # smoke-check a running API

cd frontend && npm install && npm run dev           # dashboard on :3000
npm run test && npm run build                       # 22 tests, then the bundle
```
Raw data (9.5 GB tiles) is gitignored but already on disk at `data/raw/`.


---

## Project Overview

AquaNexus is a **proof-of-concept system** demonstrating competency in:
- **HEC-RAS integration** (hydraulic/environmental simulation)
- **Machine learning** (habitat suitability prediction with explainability)
- **Full-stack development** (Python backend + React frontend)
- **Professional software engineering** (Docker, CI/CD, documentation)

**Target audience**: IGES professors, potential collaborators, recruiters  
**Deliverables**: GitHub repo (production-ready), trained model, API, web dashboard, comprehensive documentation

---

## Phase 1: Foundation & HEC-RAS Integration

**Planned**: Days 1-5  
**Actual**: 2026-09-06 → 2026-09-06  
**Status**: 🟢 Complete

### 1a. Project Setup & Architecture
- [x] Directory structure created
- [x] pyproject.toml, requirements.txt
- [x] Config management (pydantic Settings)
- [x] Docker & docker-compose setup
- [x] Initial README.md & ARCHITECTURE.md
- [x] Git repo initialized
- [x] Data source availability audit (added, not in original plan)

**Date Started**: 2026-09-05  
**Date Completed**: 2026-09-06  
**Notes**:
```
Ran a data availability audit BEFORE scaffolding (docs/DATA_SOURCES.md). It
overturned three assumptions in the original plan:

1. Study site changed: Yodo -> Ayase/Naka (Saitama). No open channel bathymetry
   exists for the Yodo. Saitama publishes river point clouds acquired by UAV AND
   narrow multibeam echosounder under CC BY 4.0 - the only open Japanese source
   found with below-water channel geometry. GSI's 5m DEM has voids exactly at
   water surfaces; 国土数値情報 W05 is centrelines only. The MLIT LAS releases
   (Tenryu, Kano, Kikugawa) are ground-classified topographic LiDAR - water
   surface, not bed - and ship under custom terms rather than CC.

2. river.go.jp cannot be scraped. It returns "This site prohibits data
   acquisition using tools" to programmatic requests. Discharge/stage must be
   pulled by hand, 30 days per request. Noted in scripts/download_data.py so
   nobody automates it later by accident.

3. Dataset design changed: designed experiment over flow space instead of a
   4-year hourly series. Water quality observations are monthly, so hourly
   nutrient/DO boundary conditions would have been interpolation artefacts.
   This invalidates the temporal validation split in ML_STRATEGY.md §5.1 -
   flagged in docs/ML_METHODOLOGY.md for revision in Phase 2.

Also: the plan justified the Yodo as "closest to IGES in Kanagawa". It is ~400 km
away in Osaka/Kyoto. Saitama is at least in Kanto.

Scaffold decisions:
- Python 3.12 local; project requires >=3.11. Dependencies split into extras
  (ml/geo/api/dashboard/dev) so the API image doesn't pull geopandas.
- settings.ensure_dirs() is explicit rather than import-time, keeping tests
  hermetic.
- Docker not installed on this machine, so Dockerfile/compose are UNTESTED.
- 15 smoke tests passing.

Ayase River carries a genuinely useful story: years as Japan's worst-BOD river,
then measurable recovery under 清流ルネッサンス. A real documented water-quality
gradient is worth more to this project than a synthetic one.
```

**Commit**: `6d4494d` Initial project structure and data source audit

---

### 1b. HEC-RAS Integration & Data Loading
- [x] Point cloud ingest (tile index, download, LAS loading)
- [x] Cross-section extraction from point cloud
- [x] HEC-RAS geometry export (.g01) + reader with round-trip test
- [x] Unit tests for hecras module (45 passing)
- [x] Docs: DATA_SOURCES.md
- [x] Water quality loader (MOE / Saitama 検体値) + download script
- [x] HEC-RAS runner (COM controller, verified against 7.0)
- [x] Steady-flow computation running end to end; depth and velocity extracted
- [x] Feature engineering (preprocessor)
- [x] Synthetic HSI label generator + falsification against real observations
- [x] Data validator (quality checks)
- [x] Notebook: 01_hecras_workflow.ipynb
- [x] Notebook: 02_data_exploration.ipynb
- [x] Docs: HECRAS_GUIDE.md

**Date Started**: 2026-09-06  
**Date Completed**: in progress  

**HEC-RAS Version**: 7.0 April 2026 (installed; COM controller verified)  
**River Selected**: 綾瀬川 Ayase, Saitama — 89 tiles, 29.4 km, fully contiguous  
**Data Source**: 埼玉県 河川点群データ (CC BY 4.0), UAV + narrow multibeam  

**Sample Output** — full pipeline, point cloud to hydraulics, tile ayasegawa-0610.
Discharges are the q01 / median / q99 of the observed Ayase record:
```
profile 2: Median  Q=9.7 m3/s
   RS   invert      WSE    depth      vel    area   width
  300     5.51     6.78     1.27   0.530    18.3    17.7
  250     5.37     6.76     1.39   0.502    19.3    17.6
  200     5.27     6.38     1.11   2.458     3.9     7.7
  150     5.19     6.43     1.24   0.767    12.7    13.6
  100     5.17     6.37     1.20   0.767    12.6    14.6
   50     5.26     6.32     1.06   0.726    13.4    15.2
```
At the high profile (64.2 m3/s) top width jumps from ~33 m to ~94 m at RS 200-150:
water spreading onto the flat terrace beside the channel. That resolves an open
question from the ingest work - the flat feature at ~7.35 m is a real flood terrace
(高水敷), not an unpenetrated water surface, since it conveys flow.

**Known data-quality issue**: RS 200 extracts anomalously narrow (1.9 m at low flow
against 10-13 m either side) and shows velocity 2.46 m/s where neighbours are ~0.5.
The section is being cut through a constriction or a structure. The data validator
should flag sections whose width departs sharply from their neighbours.

**Issues Encountered**:
- [x] **No ground classification.** Every point in the published tiles is class 1,
      so there is no `classification == 2` filter for bare earth. Resolved by
      binning across the section and taking a low elevation quantile per bin
      (`bed_quantile`, default 0.05). The minimum would be the obvious choice but
      latches onto multibeam outliers; the quantile rejects both canopy above and
      stray returns below. Covered by a regression test that adds a vegetation
      canopy and asserts the thalweg does not move.
- [x] **Axis-order ambiguity.** Japan's plane rectangular convention is X=north,
      Y=east, the opposite of GIS convention. The tiles use x=easting. Verified by
      round-tripping a tile centroid through EPSG:6677: 10 m agreement vs 16 km for
      the swapped order. Had this gone unnoticed every cross-section would have been
      misplaced by ~16 km.
- [x] **Tile index is not published as a file.** It only exists inside the Mapbox
      vector tiles behind the prefecture's web map. Recovered by decoding those tiles;
      each polygon carries MESH_NO and a direct download URL.
- [x] **Censored water quality values.** ~9,400 values per year are flagged `<`, where
      the reported number is the detection limit rather than a measurement. Treating
      those as measurements biases every statistic upward. The loader keeps the value
      as published and records the qualifier separately; `apply_censoring()` applies a
      policy (half-DL by default) as an explicit, separate step. Values flagged `>`
      are never substituted - an over-range reading is a real lower bound.
- [x] **Generated .g01 loaded as an empty model.** HEC-RAS reported zero rivers with
      no error. Four elements are required that look cosmetic: river/reach names padded
      to exactly 16 characters, a `Reach XY` centreline, and per-section `XS GIS Cut
      Line` / `#Mann` / `Bank Sta` / `XS Rating Curve` / `Exp/Cntr` blocks. Verified
      fixed through the COM controller - 1 river, 6 sections, correct station order.
- [x] **The plan selects the geometry, not the project.** A .prj naming `Geom File=g01`
      still gives an empty CurrentGeomFile until a .p01 exists and `Current Plan` names it.
- [x] **A dangling sidecar reference hangs COM automation.** `Unsteady File=u01` with no
      .u01 on disk raises a modal dialog; over COM the call blocks forever with no error
      and leaves orphaned Ras.exe processes. `validate_project()` now checks for this
      before opening anything, and RasController kills surviving processes on exit.
- [x] **Bank stations must be values that exist in the station/elevation list.**
      The obvious "35% of the way across" rule produces an interpolated station, and
      extraction drops sparse bins so the list has gaps. HEC-RAS rejected every section
      with "Left bank station not in station elevation data" and refused the run.
      Banks are now found by walking outward from the thalweg, which returns real
      stations by construction. Manning break points had the same defect.
- [x] **A hand-written plan file cannot work.** HEC-RAS 7.0's own plan carries ~200
      settings; a minimal one is rejected with only "there must have been some missing
      data in the input files", naming neither file nor field. The plan and project are
      now templates captured from a reference project HEC-RAS wrote and computed itself.
- [x] **Double CRLF from Python's text layer.** Writing an already-CRLF string without
      `newline=""` emits 


. HEC-RAS accepts the file, then hangs on a modal
      dialog rather than reporting a parse error.
- [x] **The detailed diagnosis is written to `<plan>.computeMsgs.txt`.** The COM
      controller returns only the generic summary; the per-station reasons are on disk.
      Four blind attempts were spent before finding that file.
- [x] **Windows console mangled Japanese log output** (綾瀬川 printed as escapes).
      Logger now reconfigures the stream to UTF-8, guarded for detached streams.

**Observed data loaded** (3 fiscal years, 3,892 samples; Ayase = 216 across 5 stations):
```
      water_temp  discharge  DO    BOD   TN    TP
2022       21.98      21.60  5.93  2.11  2.82  0.19
2023       18.80      14.41  7.30  2.85  3.41  0.24
2024       19.47      16.89  6.70  2.55  3.14  0.22

DO vs temperature:  >25 C -> 5.13 mg/L (n=59)   <15 C -> 8.74 mg/L (n=72)
```
The thermal-oxygen coupling is present in the real observations, which is the
relationship the habitat model is supposed to reproduce. It also gives a genuine
check on the synthetic labels: if the generated HSI does not degrade under the
warm/low-DO conditions seen here, the label function is wrong.

**Commit**: `6d4494d` Initial project structure and data source audit

---

### 1c. Data Exploration & Validation
- [x] Real Japanese river data loaded (3 fiscal years, 3,892 samples)
- [x] HEC-RAS simulation completed (full 27.5 km reach, 3 flow profiles)
- [x] Data quality report generated (validate_sections: 0 errors, 20 warnings)
- [x] Data saved to data/processed/ayase_hydraulics.csv
- [x] EDA notebook completed
- [→] Environmental state vectors — **deliberately moved to Phase 2**

**Scope change (2026-09-06):** "Environmental state vectors" is listed under Phase 1c
in AQUANEXUS-PLAN.md, but it is not a data-pipeline deliverable — it is the modelling
dataset. Building it means choosing how simulated hydraulics join to monthly
observations, which features enter the matrix, and how the design grid is sampled.
Those are ML design decisions governed by ML_STRATEGY.md §3 and §5, so the work
belongs with Phase 2a where those choices are made and recorded. Nothing was dropped;
the item moved to where its decisions live.

**Full reach result** (89 tiles, 9.5 GB, 500 m spacing):
```
centreline 27.5 km -> 56 candidates -> 53 sections after de-duplication
compute succeeded for all three profiles
  Low    (Q=0.25)  depth 0.04-5.15 m
  Median (Q=9.7)   depth 0.23-6.05 m
  High   (Q=64.2)  depth 1.38-7.67 m
bed falls 10.8 m over 27.5 km = 0.39 m/km, correct sign, plausible lowland slope
```

**Quality**: 0 errors, 20 warnings. Four sections cut through constrictions or
structures (RS 12500, 14000, 18000, 24000 - widths of 11-37 m against neighbours
of 65-131 m); fifteen show bed steps over 2 m between adjacent sections; one is
sparse. Median adjacent invert step is 0.53 m and 35 of 52 steps move in the
expected direction, so the reach is coherent overall with local noise where the
centreline (derived from tile centroids) wanders off the channel. Those sections
should be excluded or re-cut before the geometry is used for anything load-bearing.

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**Data Statistics**:
```
Water quality:   3,892 samples over 3 fiscal years (FY2022-24), 101-102
                 parameters per file, 2022-04-06 to 2025-03-12
  Ayase subset:  216 samples, 5 stations; 138 carry both discharge and DO
  Censored:      ~9,400 values/year flagged "<" - kept as published, with the
                 qualifier recorded separately rather than silently treated
                 as measurements
Point cloud:     89 tiles, 9.5 GB, contiguous over 29.4 km
Geometry:        53 cross-sections, 27.5 km, bed falling 0.39 m/km
Flow sweep:      53 sections x 12 log-spaced discharges (0.17-73.7 m3/s)
State vectors:   7,314 rows x 35 columns (53 sections x 138 observations)
```

**Features Created**: 12 derived, in `aquanexus.data.preprocessor` - DO
saturation (Benson-Krause, tested to +/-0.03 mg/L against the published table),
DO deficit and percent saturation, Froude, Reynolds, shear stress, thermal and
oxygen stress indices, and three interaction terms. Lagged and rolling features
from ML_STRATEGY §3.2 are deliberately **not** produced: they assume an hourly
chronology this dataset does not have.

**Commit**: `62decfb` Add point cloud ingest and cross-section extraction · `f578182` Run HEC-RAS end to end

**Phase 1 Summary**:
```
✓ Completed: [What was accomplished]
✓ Data pipeline: Fully functional
✓ Ready for: ML training
⚠️ Challenges: [Any issues that might affect later phases]
→ Next: Begin Phase 2 (ML pipeline)
```

---

## Phase 2: Machine Learning Pipeline

**Planned**: Days 6-12  
**Actual**: 2026-09-06 → 2026-09-07  
**Status**: 🟢 Complete

### 2a. Model Architecture & Training
- [x] Environmental state vectors (moved from 1c) — 7,314 rows, 53 sections x 138 obs
- [x] Flow sweep: 12 log-spaced discharges spanning the observed range
- [x] Split strategies (grouped, spatial, flow, temporal, + leaky random for contrast)
- [x] Model set: XGBoost, Random Forest, Ridge, mean-predictor floor
- [x] Benchmark across all splits

**Second target added — observed dissolved oxygen (real labels).** R² 0.442,
RMSE 1.71 mg/L under grouped CV with each station held out. Ridge beats both tree
models at n=138. The DO-saturation baseline scores R² −0.60 with bias +2.39 mg/L,
i.e. the Ayase runs a persistent ~2.4 mg/L oxygen deficit — a real measured
property. Ablation: raw discharge *hurts* (−0.019 R²) while the hydraulic model's
transformation of it helps (+0.062), so the physics-informed step carries
information the raw driver does not.

**Synthetic HSI result**: R² 0.99 on every split — which is a finding about the labels,
not a modelling success. HSI is a deterministic, noiseless function of the features,
so the model recovers an analytic function rather than learning ecology. A depth-4
tree on depth alone already reaches R² 0.66; depth+velocity reaches 0.88. Recorded
in docs/ML_METHODOLOGY.md with the full diagnosis.
- [x] Model factory (XGBoost, Random Forest, Ridge, mean floor — no LSTM, see note)
- [x] Training pipeline with validation
- [x] Model serialization (pickle/joblib)
- [ ] Hyperparameter tuning (optional — skipped, n=138 does not support it)
- [x] Training notebook (03_model_training.ipynb)
- [x] Unit tests for models

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**Model Configuration**:
```python
model_type: xgboost
n_estimators: 200
max_depth: 7
learning_rate: 0.05
random_seed: 42
train_size: 0.7
val_size: 0.15
test_size: 0.15
temporal_split: True
```

**Training Results**:
```
Synthetic HSI (xgboost, grouped split - whole observations held out):
  Train 4,717 (64%) | Val 1,113 (15%) | Test 1,484 (20%)
  RMSE 0.024   MAE 0.015   R2 0.993   skill vs mean 0.993

Observed DO (Ridge, grouped CV - each of 4 stations held out in turn):
  138 rows, 10 features, no holdout to spare - every row is scored
  out-of-fold
  RMSE 1.713   MAE 1.232   R2 0.442   Pearson 0.680   NSE 0.442

scripts/train_models.py end to end: 6.0 s
Artefacts: hsi_v1.joblib 674 KB, dissolved_oxygen_v1.joblib 2.5 KB,
           plus a 100-row SHAP background per model
```

**Model Performance**: the two rows above are the whole finding. R2 0.993 on
generated labels is function recovery; R2 0.442 on measured labels is the
result. Reporting the first without the second would be the single most
misleading thing this project could do.

**Commit**: `2464989` Add data validator, notebooks and HEC-RAS guide · `f734d0f` Build and run the full 27.5 km reach

---

### 2b. Explainability & Threshold Detection
- [x] SHAP explainer with model-type dispatch (Tree/Kernel)
- [x] Feature importance by mean |SHAP|, signed and absolute
- [x] Interaction analysis with identifiability reporting
- [x] Threshold discovery (marginal response sweeps)
- [x] Summary plot + saved SHAP values
- [x] Collinearity detection

**Key result — RETRACTED 2026-09-07**: this said water_temp x discharge interact
synergistically at -1.02 mg/L. That run explained a 48-row subset, which is one
station (52内匠橋). Over all 138 observations the interaction is +0.23 mg/L and the
sign flips per station (-0.93 to +1.72). Not established at this sample size. Full
working in 04_explainability.ipynb; docs/ML_METHODOLOGY.md and README corrected.

**Key limitation found**: 10 feature pairs correlate at |r| >= 0.9 over all 138
rows (12 over that 48-row subset — the count moves with what is explained), so individual
SHAP ranks are not trustworthy - the hydraulic features are all derived from
discharge. Two of four interaction pairs are *unidentifiable* (an empty corner in
the 2x2 design). The explainer reports that rather than returning NaN.
- [x] SHAP explainer implemented
- [x] Feature importance ranking
- [x] Interaction analysis
- [x] Critical thresholds discovered
- [x] Explainability notebook (04_explainability.ipynb)

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**Feature Importance** (mean |SHAP|, dissolved oxygen, all 138 rows):
```
Ranked, but NOT rankable: 10 feature pairs correlate above 0.9, so SHAP
divides credit between them arbitrarily. Read the hydraulic features as one
combined contribution.

do_saturation, reach_froude, reach_depth, month_sin, air_temp, discharge,
reach_velocity, reach_top_width, month_cos, water_temp
```

**Critical Thresholds Discovered** (marginal response across the observed range,
everything else at its median):
```
do_saturation    span 5.42 mg/L   steepest at 8.22 mg/L   influential
reach_top_width  span 3.62 mg/L   steepest at 39.7 m      influential
air_temp         span 2.33 mg/L   steepest at 6.8 °C      influential
reach_froude     span 1.90 mg/L   steepest at 0.039       influential
reach_depth      span 1.70 mg/L   steepest at 2.04 m      influential
discharge        span 1.53 mg/L   steepest at 16.6 m3/s   influential
water_temp       span 0.32 mg/L                           NOT influential
```
`water_temp` looks inert only because the model routes the thermal signal
through `do_saturation`, which is a deterministic function of it (r = -0.99).
Temperature matters most of all; this is the collinearity caveat made visible.

**Interaction Findings**: of four pairs tested, two are unidentifiable (an empty
corner in the 2x2 design, because the features correlate at 0.89-0.99) and the
explainer reports that rather than returning NaN. The one headline result -
water_temp x discharge synergistic at -1.02 mg/L - was **retracted on 09-07**:
it came from a single station's 48 rows, and pooled over all 138 the sign
reverses (+0.23, per-station range -0.93 to +1.72). Not established at this
sample size.

**Commit**: `aaeb189` Phase 2a: state vectors, splits, models and benchmark · `5d8d652` Add dissolved oxygen target

---

### 2c. Model Validation & Baselines
- [x] Three specified baselines (hydraulic-only, linear, persistence)
- [x] Spatial validation (per-station holdout)
- [x] Event-based validation (discharge/temperature tails)
- [x] Comparison report table

Model beats persistence by only 0.06 R² and loses on MAE. Under-predicts DO by
2 mg/L at low flow — worst exactly where it matters. Hydraulic-only scores below
the mean. Full numbers in docs/ML_METHODOLOGY.md.
- [x] Temporal validation (train/val/test time split)
- [x] Spatial validation (train on some reaches, test on others)
- [x] Event-based validation (normal vs extreme conditions)
- [x] Baseline comparisons (hydraulic-only, linear, persistence)
- [x] Validation notebook (05_model_validation.ipynb)

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**Validation Results Table** (grouped CV, each station held out, observed DO):
```
| Model            | RMSE  | MAE   | R²     | Verdict                          |
|------------------|-------|-------|--------|----------------------------------|
| Ridge (linear)   | 1.713 | 1.232 |  0.442 | ✅ Selected — best RMSE           |
| Persistence      | 1.818 | 1.213 |  0.385 | ⚠ Beats the model on MAE         |
| Random forest    | 1.879 | 1.417 |  0.329 | ✗ Overfits at n=138              |
| XGBoost          | 1.904 | 1.409 |  0.311 | ✗ Overfits at n=138              |
| Mean (floor)     | 2.294 | 1.779 |  0.000 | — the floor                      |
| Hydraulic-only   | 2.467 | 1.862 | -0.157 | ✗ Worse than guessing the mean   |
| DO saturation    | 2.900 | 2.577 | -0.598 | ✗ Right shape, +2.39 mg/L bias   |
```
The plan predicted XGBoost 0.84 and linear 0.62. Both are wrong in both
directions: the tree models lose, and nothing gets near 0.84 on real labels.

**Temporal Generalization**: not applicable. The dataset is a designed
experiment over flow space, not a chronology — there is no "future" to hold out.
`temporal_split()` exists for the observations, which do carry dates, but that is
a weaker claim than ML_STRATEGY §5.1 makes and is not what the shipped model is
validated on.

**Spatial Generalization** (the real version — whole stations held out):
```
55畷橋            n=36   observed mean 8.36   RMSE 2.206   bias -1.540
54槐戸橋          n=36   observed mean 7.39   RMSE 1.793   bias -0.444
57綾瀬川合流点前   n=18   observed mean 6.27   RMSE 1.439   bias +1.205
52内匠橋          n=48   observed mean 6.15   RMSE 1.266   bias +0.115
```
The upstream station is the hard one: it is shallow, low-flow and oxygen-rich,
and the model pulls it toward the reach mean.

**Event-based Generalization** (tails of the drivers, standing in for events):
```
low flow  (≈1.1 m3/s)   n=21   RMSE 2.532   bias -2.011
middle    (≈13 m3/s)    n=96   RMSE 1.634   bias -0.043
high flow (≈52 m3/s)    n=21   RMSE 0.818   bias +0.103
```
**The −2 mg/L low-flow bias is the headline limitation of the whole project.**

**Baseline Comparison**: the model beats persistence by 0.105 mg/L RMSE and
0.057 R², and **loses to it on MAE** (1.232 vs 1.213). It earns its place by
generalising to unseen stations and by accepting hypothetical states — not by
being much more accurate than "same as last month".

**Phase 2 Summary**:
```
✓ Completed: Trained XGBoost model (R²=0.84)
✓ Explainability: SHAP framework integrated
✓ Thresholds: Critical conditions identified
✓ Validation: Temporal, spatial, event-based splits passed
✓ Model file: data/models/xgboost_v1.joblib
✓ Ready for: Backend API development
```

---

## Phase 3: Backend API

**Planned**: Days 13-16  
**Actual**: 2026-09-06 → 2026-09-07  
**Status**: 🟢 Complete

### 3a. API Design & Core Endpoints
- [x] FastAPI app skeleton
- [x] Pydantic request/response schemas
- [x] POST /predict endpoint
- [x] POST /batch_predict endpoint
- [x] GET /health endpoint (plus GET /models, POST /explain, POST /scenario_run)
- [x] Swagger/OpenAPI docs auto-generated (/docs, /openapi.json)
- [x] Unit tests for routes — 24, plus 24 integration tests added in 3b

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**API Endpoints Implemented** (all seven — see docs/API_REFERENCE.md):
```
GET  /                service description
GET  /health          liveness + whether models loaded; "degraded" with a reason
GET  /models          provenance, metrics, caveats, training ranges per model
POST /predict         value + interpretation + caveats + out-of-range flags
POST /batch_predict   1–1000 states; a bad row returns NaN rather than voiding
POST /explain         SHAP contributions + the collinear pairs that make them
                      unrankable
POST /scenario_run    fractional what-ifs against a baseline
```
Departures from the original sketch: no CSV upload (JSON throughout), no
`/explain/{id}` (nothing is stored, so there is no id), and no
`/interactive_analysis` — the dashboard's Analyze page is `/batch_predict` over
a grid, which needed no new endpoint.

**Example Request/Response** (real, from a running instance):
```json
// POST /predict
{"target": "dissolved_oxygen",
 "state": {"water_temp": 24.5, "discharge": 12.0, "depth": 2.9,
           "velocity": 0.41, "month": 7}}

{"target": "dissolved_oxygen",
 "prediction": 5.83,
 "unit": "mg/L",
 "interpretation": "5.83 mg/L - adequate for most species",
 "labels": "observed",
 "uncertainty": null,
 "out_of_range": [],
 "caveats": ["Labels are real measurements from the 公共用水域 monitoring record.",
             "Beats a persistence baseline by only 0.06 R2 and loses to it on MAE.",
             "..."]}
```

No `confidence` field: Ridge cannot express a spread, so `uncertainty` is null
rather than a fabricated number. No `timestamp` echo - the caller already has it.

**API Testing**: `python scripts/verify_deployment.py` - 14 checks, all passing.
Model loading ~60 ms, warm prediction ~3 ms, warm explanation ~190 ms.

**Commit**: `9d3266d` Phase 3a: FastAPI backend serving both models with provenance

---

### 3b. Integration & Deployment
- [x] Trained model loaded on startup
- [x] Feature preparation pipeline integrated
- [x] HEC-RAS scenario runner integrated (sweep interpolation; see the caveat below)
- [x] Error handling & logging
- [x] Middleware (CORS, request logging)
- [x] Dockerfile created
- [x] docker-compose.yml created
- [x] Environment configuration (.env)
- [x] Full API documentation (docs/API_REFERENCE.md)
- [x] Integration tests (21) + deployment invariant tests (14)
- [x] `scripts/verify_deployment.py` — 14 smoke checks against a live service
- [ ] `docker build` / `docker compose up` — **no Docker on this machine**

**Three defects found and fixed while verifying**

1. **`/explain` returned zero for every feature.** The SHAP explainer was fitted
   on the request row itself, so the prediction was measured against a
   background of one identical point: baseline == prediction, all contributions
   exactly 0.0. Well-formed, correctly summing, and empty. The training sample
   now ships as `<target>_background.csv` and the explainer is built once per
   model at first use. Warm explain latency 2.4 s → 190 ms as a side effect.
2. **`collinear_pairs` was documented in the schema and never populated.** The
   route read an attribute that did not exist and returned `[]` every time, so
   the one caveat that keeps a caller from ranking ten collinear features was
   silently absent. Computed at training time now, served with every explanation.
3. **`DATA_DIR` resolved into site-packages for an installed package.** Settings
   derive paths from the source file location, which in the image is
   site-packages, not `/app` — so `-v ./data:/app/data` would have changed
   nothing and the container would have started degraded with a correct-looking
   mount. `DATA_DIR` is pinned in the Dockerfile and now propagates to the
   subdirectories.

Plus one that would have wasted somebody's afternoon: compose mounted `./src`
over an image that installed the package non-editably, so the mount was inert
and `--reload` restarted on edits that could not take effect. Fixed with
`PYTHONPATH=/app/src`.

**Integration Status** (Windows, Python 3.12, single worker)
```
Model loading (both models):        ~60 ms
Explainer init (first /explain):    ~2.4 s, once per model, lazy
Warm prediction latency:            ~3 ms
Warm explanation latency:           ~190 ms (DO), ~220 ms (HSI)
Batch of 100 states:                ~1 prediction's cost
```

**Docker Build**: NOT RUN — Docker is not installed here. Verified instead:
```
pip install ".[ml,api]" into a clean venv     PASS  (the image's install step)
serve from the installed copy, no src on path PASS  (mirrors the image layout)
DATA_DIR unset -> degraded, /health explains  PASS  (mirrors a missing volume)
DATA_DIR set   -> 14/14 smoke checks pass     PASS  (mirrors compose)
tests/test_deployment.py                      14 tests on Dockerfile + compose
```
Still unverified without a daemon: the base image, apt/libgomp, Linux wheels,
the non-root user against a bind-mounted volume, and the HEALTHCHECK loop.

**Whoever has Docker runs:**
```bash
docker compose up -d
python scripts/verify_deployment.py --url http://localhost:8000
```

**Commit**:
```
Phase 3b: integration tests, deployment verification, three fixes
```

**Phase 3 Summary**:
```
✓ Completed: FastAPI backend fully functional
✓ Endpoints: /predict, /batch_predict, /health working
✓ Docker: Multi-stage build created, tested locally
✓ Documentation: API_REFERENCE.md complete
✓ Ready for: React frontend integration
```

---

## Phase 4: Frontend & Deployment

**Planned**: Days 17-28  
**Actual**: 2026-09-07 → 2026-09-07  
**Status**: 🟢 Complete — dashboard built and driven in a browser; images unbuilt

### 4a. React Frontend
- [x] React project initialised (Vite + TypeScript, React 19)
- [x] API client configured — `fetch`, not axios (see the note below)
- [x] Dashboard (Home) with both models side by side
- [x] EnvironmentalInput form component
- [x] PredictionCard component
- [x] ScenarioBuilder component
- [x] ExplainabilityPanel component (SHAP bars, inline SVG)
- [x] InteractionHeatmap component (81-call response surface)
- [x] Pages: Home, Predict, Scenarios, Analyze, About
- [x] Styling — hand-written CSS, not Material-UI or Tailwind (see below)
- [x] Responsive design (single-column below 860 px; charts scroll in their own box)

**The design constraint that shaped everything.** One of the two served models is
trained on labels this project generated. An interface that renders a habitat score
as a confident number with the caveat one click away would undo what the rest of the
repo is careful about. So the provenance badge sits in the same card as the number,
the API's caveats are rendered with the prediction rather than summarised, and the
collinearity warning is rendered *above* the SHAP chart — the bar order is not a
ranking, and a reader who sees the chart first has already concluded that it is.
`src/test/provenance.test.tsx` asserts those properties so a refactor cannot drop them.

**Two deliberate deviations from the plan.** §4a specifies axios and a UI kit. The
API client is ~100 lines of request building, so a dependency there would only need
mocking in every test; and a dozen components do not need a framework. The CSS tokens
are the same ink/accent/cool used by `scripts/make_figures.py`, so the app and the
README figures read as one project.

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**API Integration Test** — driven in a real Chrome, not just jsdom:
```
Frontend http://localhost:3000  ->  Backend http://localhost:8002

✓ Overview loads both models with correct provenance badges and live metrics
✓ Predict -> 5.83 mg/L with the four caveats attached
✓ Predict and explain -> SHAP bars, 10 collinear pairs disclosed above the chart
✓ Analyze -> 81 predictions in one batch call, heatmap renders
✓ Degraded backend -> banner naming the fix, verified by pointing at a dead port
```

**Three things the browser found that the tests did not:**
1. **CORS.** Opening the dev server at `127.0.0.1:3000` failed with an
   unexplained "cannot reach the API" — the allowlist had only the `localhost`
   spelling, and a browser treats them as different origins. Both are listed now.
2. **The default state opened out of range.** Depth 1.8 m is below the training
   minimum for `reach_depth` (1.86), so the app loaded with a red warning already
   showing. Defaults moved to the reach means at 12 m³/s.
3. **The API and the UI disagreed about range checks.** The UI flagged that depth
   while `/predict` did not: `build_features` maps a caller's point `depth` onto
   `reach_depth`, but `check_ranges` looked only for the literal name, so an
   aliased value reached the model unflagged. The check follows the aliases now.

**Bundle**: 262 kB (83 kB gzipped), 38 modules, ~0.5 s build.

**Commit**: `1708e87` Phase 2c: validation, baselines and a short-form daily log

---

### 4b. Testing, Polish & Deployment
- [x] Backend unit tests (pytest) — 355
- [x] Backend integration tests — 24, added in Phase 3b
- [x] Frontend component tests (React Testing Library) — 22
- [x] E2E: state → prediction → explanation, in a real browser against the real API
- [x] UI/UX polish — shared palette with the figures, labelled controls, responsive
- [x] Docker multi-stage build — frontend (node → nginx); API stays single-stage
- [x] GitHub Actions CI/CD pipeline — backend matrix, frontend, container builds
- [x] Final documentation (README, DEPLOYMENT, API_REFERENCE, frontend README)
- [ ] `docker build` still unrun locally — no daemon; the CI `containers` job is
      the first place either image is actually built

**Date Started**: 2026-09-06  
**Date Completed**: 2026-09-07  

**Test Coverage**:
```
Backend:  355 passing (pytest), ruff clean
          - 24 integration, 20 deployment invariants, rest unit
Frontend:  22 passing (vitest + React Testing Library)
          - 9 provenance contract, 5 API client, 8 app flows
```

**CI/CD Pipeline** (`.github/workflows/ci.yml`, push + PR):
```
backend     Python 3.11 and 3.12 · ruff check · pytest
frontend    npm ci · npm run build (typecheck + bundle) · npm run test
containers  docker build both images · start the API · verify_deployment.py
```
The `containers` job exists because this machine has no Docker: it is the first
place the images are built at all. The smoke step is `continue-on-error` since CI
has no trained artefacts, so the API starts degraded there by design.

**Production Docker Image**: not built locally. The frontend image is the
multi-stage one (node build → nginx serve, with an SPA fallback so a reload on
/predict does not 404). The API image stays single-stage: its heavy dependencies
are runtime dependencies, so a build stage would save nothing.

**Documentation Status**:
```
✓ README.md            overview, results, dashboard, honest limitations
✓ docs/ARCHITECTURE.md module boundaries
✓ docs/API_REFERENCE.md endpoints, provenance, performance (was a placeholder)
✓ docs/DEPLOYMENT.md   running it + a verification-status section
✓ docs/ML_METHODOLOGY.md every result and deviation
✓ docs/DATA_SOURCES.md  availability audit
✓ docs/HECRAS_GUIDE.md  file-format traps
✓ frontend/README.md    dashboard structure and its constraints
✓ LICENSE              MIT, added here — pyproject had declared it since day one
✓ CONTRIBUTING.md      how to run it, and the conventions a change should follow
```

**Deployment Options**: `docker compose up` brings up both services (API 8000,
dashboard 3000). Cloud targets are deliberately not documented as recipes —
nothing here is production-hardened, and `docs/DEPLOYMENT.md` lists what would
have to change first (no auth, no TLS, no rate limiting, artefacts mounted from
the host).

**Final Commit**: `f7b2d7d` Phase 4: React dashboard, container build, CI

**Phase 4 Summary**:
```
✓ Completed: Full-stack system ready for production
✓ Frontend: React dashboard with all features
✓ Backend: FastAPI with comprehensive documentation
✓ Testing: Unit, integration, E2E tests passing
✓ Deployment: Docker images built, CI/CD active
✓ Documentation: Complete and professional
✓ GitHub: Production-ready repo with clean history
```

---

## Project Completion Checklist

Ticked against what was actually built, with the original target kept where it
turned out to be the wrong target.

- [x] **Phase 1**: Data pipeline complete — ~~50K+ records~~ **3,892 water
      quality samples and 9.5 GB of point cloud**. The 50K figure assumed an
      hourly series that does not exist for any open Japanese river; the audit
      that established this is `docs/DATA_SOURCES.md`.
- [x] **Phase 2**: Model trained, explainability working — ~~R² ≥ 0.80~~
      **R² 0.442 on real labels**. The 0.80 target was set against synthetic
      labels, where this project scores 0.993 and the number means nothing.
      0.442 on measured oxygen is the honest result and it is a modest one.
- [x] **Phase 3**: FastAPI backend fully functional — seven endpoints, 48 tests.
      **Docker ready but unbuilt**: no daemon on this machine.
- [x] **Phase 4**: React frontend, tests passing, CI/CD configured — 22 frontend
      tests, workflow on push and PR. The CI has never run; the first run is
      also the first time either image is built.
- [x] **Documentation**: README, ARCHITECTURE, API_REFERENCE, DEPLOYMENT,
      ML_METHODOLOGY, DATA_SOURCES, HECRAS_GUIDE, frontend README, 5 notebooks
- [x] **GitHub**: clean history, 24 commits, each explaining why
- [x] **Model files**: ~~xgboost_v1.joblib~~ **dissolved_oxygen_v1.joblib +
      hsi_v1.joblib + manifest.json + a SHAP background each**. Two models, not
      one, because one of them is trained on generated labels and shipping only
      that would have been the dishonest choice.
- [x] **DEVLOG.md**: complete chronicle, including the retraction
- [ ] **Production ready**: `docker compose up` → **unverified**. It is the one
      claim in this list that cannot be made from this machine, and it is not
      going to be made without evidence.

## Key Metrics & Results

**Data Pipeline**
```
River:         Ayase (綾瀬川), Saitama — 27.5 km, 53 cross-sections
Records:       3,892 water quality samples; 216 on the Ayase; 138 modelled
Features:      26 (synthetic HSI dataset) / 10 (observed DO dataset)
Time period:   2022-04-06 to 2025-03-12 (FY2022-24)
Completeness:  138 of 216 Ayase samples carry both discharge and DO (64%)
               ~9,400 censored values per year, kept as published
```

**Model Performance** (grouped CV, whole stations held out — observed DO)
```
RMSE: 1.713 mg/L        Pearson: 0.680
MAE:  1.232 mg/L        NSE:     0.442
R²:   0.442             Skill vs mean: 0.442

Temporal generalisation: N/A — designed experiment, no chronology to split
Spatial generalisation:  ✓ tested, 4 stations held out in turn (RMSE 1.27–2.21)
Event generalisation:    ⚠ tested and FAILS at low flow (bias −2.01 mg/L)
```

**System Performance**
```
Model loading (both):        ~60 ms
Warm prediction:             ~3 ms
Batch of 100 states:         ≈ the cost of one prediction
First explanation per model: ~2.4 s (builds the SHAP explainer, once)
Warm explanation:            ~190 ms
Frontend bundle:             262 kB, 83 kB gzipped, ~0.4 s build
```

**Code Quality**
```
Tests:      355 backend (pytest) + 22 frontend (vitest)
Lint:       ruff clean across src, tests, scripts
Docs:       8 documents + 5 executable notebooks
Commits:    24, each stating why rather than what
```

## Important Decisions & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| River | Ayase (綾瀬川) | The only open Japanese source found with *submerged* channel bathymetry (UAV + multibeam, CC BY 4.0). The planned Yodo has none. |
| Second target | Observed dissolved oxygen | The synthetic HSI made the whole project circular. A measured label was the only way to test anything. |
| Model | **Ridge**, not XGBoost | At n=138 gradient boosting overfits and cross-validates worse. The plan assumed XGBoost throughout; the data disagreed. |
| Both models shipped | Yes, each labelled | Serving only the 0.99 model would mislead; serving only the 0.44 model would drop the habitat framing. |
| Explainability | SHAP with model-type dispatch | `TreeExplainer` cannot explain a Ridge pipeline, which is what actually ships. |
| Splits | Grouped, always | 53 rows share one observation's chemistry; an ungrouped split scores memorisation. |
| Backend | FastAPI | Async, typed, self-documenting; the schema *is* the contract the frontend imports. |
| Frontend | React + TS, hand-written CSS | A dozen components do not need a UI kit, and the palette is shared with the figures. |
| API client | `fetch`, not axios | ~100 lines of request building; a dependency would need mocking everywhere. |
| Deployment | Docker + compose | Reproducible — though unverified here, and said so. |

## Challenges & Solutions

### Challenge 1: the planned river had no usable data
**Problem**: The Yodo has no open channel bathymetry, and `river.go.jp` blocks
automated access to the national hydrology database.
**Solution**: A data availability audit *before* scaffolding, which moved the
study site to the Ayase and changed the dataset design from a time series to a
designed experiment over flow space.
**Result**: ✓ RESOLVED — `docs/DATA_SOURCES.md`. Cost a day, saved the project.

### Challenge 2: HEC-RAS geometry that fails silently
**Problem**: Generated `.g01` files were accepted and produced nothing usable.
**Solution**: Bank stations must appear in the station list, and the plan file
needs ~200 keys — templated from a real project rather than guessed.
**Result**: ✓ RESOLVED — 53 sections, 3 profiles, bed falling 0.39 m/km.

### Challenge 3: the machine learning was circular
**Problem**: HSI labels are generated by this repository, so R² 0.99 measured
how well a regressor recovers a formula the project itself wrote.
**Solution**: Added a second target with **measured** labels (observed DO), and
a falsification test for the synthetic ones.
**Result**: ✓ RESOLVED — and the honest number, 0.442, is the one now reported.

### Challenge 4: a published finding that did not hold
**Problem**: A temperature × discharge synergy of −1.02 mg/L was reported in
Phase 2b, in the README and the methodology doc.
**Solution**: Rebuilding it for the notebooks showed it came from one station's
48 rows; pooled over all 138 the sign reverses.
**Result**: ✓ RETRACTED — corrected in three documents rather than quietly
dropped.

### Challenge 5: an explanation endpoint that explained nothing
**Problem**: `/explain` returned 0.0 for every feature. The SHAP background was
the request row itself, so baseline equalled prediction. It was well-formed,
summed correctly, and said nothing — and a test asserting the sum passed on
0 == 0.
**Solution**: Ship a training-set background beside each model; build the
explainer once per model.
**Result**: ✓ RESOLVED — and warm explanations went from 2.4 s to 190 ms.

## Lessons Learned

1. **Audit the data before writing the code.** Three of the plan's assumptions —
   the river, the discharge range, the record length — were wrong, and one day
   of checking sources saved rebuilding on a foundation that did not exist.
2. **A number without provenance is worse than no number.** Two models with
   different label provenance made every downstream decision clearer: what to
   test, what to serve, what to put on a badge in the UI.
3. **Baselines are the finding.** Persistence beats this model on MAE. Without
   that row the R² of 0.442 would have read as competence.
4. **Tests can pass on nothing.** The contributions-sum test passed while every
   contribution was zero. Assert that a thing is non-degenerate, not just
   self-consistent.
5. **Run the app.** Three defects — CORS, out-of-range defaults, and the
   alias mismatch between UI and API — survived 377 passing tests and appeared
   within two minutes in a browser.
6. **Quantify what you cannot fix.** Manning's n could not be calibrated, so it
   sat in the limitations as "unquantified systematic error" for three days.
   Re-running the sweep across the plausible range took twenty minutes and
   turned it into a number. It also had to be re-run after the geometry was
   corrected, which moved it from 64% of the model's error to 15% — the first
   answer was mostly the bad cross-sections talking. An unquantified limitation
   is an unranked one, and a study is only as current as the artefacts under it.
7. **For next version**: a gauged rating curve first — the sensitivity study says
   so — then re-cut the four bad cross-sections, then low-flow observations.

## Next Steps (Beyond PoC)

- [ ] Integrate real biological validation data
- [ ] Deploy to cloud (AWS, Google Cloud, Heroku)
- [ ] Add user authentication & database
- [ ] Expand to multiple Japanese rivers
- [ ] Publish paper/results
- [ ] Community feedback & iterations

---

## Final Notes

**Project Status**: 🟢 **COMPLETE** — with one claim outstanding: neither
container image has been built, because this machine has no Docker daemon. CI
builds both on the first run.

This proof of concept demonstrates:
- **HEC-RAS integration** — driven through its COM automation server, from raw
  bathymetric point clouds to a 53-section, 27.5 km model and a 12-discharge
  sweep
- **A machine learning pipeline that argues with itself** — two targets, one
  synthetic and one measured, grouped cross-validation, baselines that the model
  only just beats, and a finding retracted when it did not survive re-analysis
- **Full-stack delivery** — FastAPI serving provenance and caveats with every
  number, and a React dashboard built so the caveats cannot be separated from
  the number
- **Professional practice** — 377 tests, ruff clean, CI on both halves, eight
  documents, five executable notebooks, and a debt register that travels with
  the analysis

**What it does not demonstrate**: ecological skill. The habitat index is trained
on labels this repository generated, and the oxygen model beats "same as last
month" by 0.06 R² and fails at low flow. Both are stated everywhere they appear,
which is the point.

**Recommended for**: portfolio, GitHub showcase, IGES discussion, and as a
worked example of reporting a modest result honestly.

**Last Updated**: 2026-09-07  
**Last Commit**: `f7b2d7d` Phase 4: React dashboard, container build, CI  

---

**Developer Signature**: kiruthick01  
**Date**: 2026-09-07
