# AquaNexus Development Log

**Project**: AquaNexus — Physics-Informed ML Framework for Aquatic Ecosystem Diagnosis  
**Developer**: kiruthick01  
**Timeline**: 2026-09-05 → in progress  
**Status**: 🟡 In Progress  
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


### Next session — pick up here

**State:** All four phases done. 355 backend tests + 22 frontend, lint clean, all pushed.

**Next, in order of value:**
1. Watch the first CI run: the `containers` job builds both images for the first time. Expect it to be where the remaining Docker unknowns surface.
2. Cheap and worth it: re-derive reach hydraulics from the sweep inside `/scenario_run` (see below).
3. Optional polish: dark mode; a shareable permalink for a state; caching `/explain` by state, since it is ~200 ms of SHAP per call.

**Known debt:**
- 4 cross-sections cut through constrictions (RS 12500/14000/18000/24000) — flagged by validator, not excluded.
- Manning's n uncalibrated — depth/velocity carry unquantified systematic error.
- `/scenario_run` holds depth/velocity/width fixed when discharge changes, so every discharge scenario describes a state the river cannot be in. Fix: re-interpolate from `ayase_flow_sweep.csv` (worked example in 05 §6).
- Scenario answers below ~2 m³/s should not be believed regardless of that fix — the model is biased −2 mg/L there and drought mechanisms (heat, residence time, concentrated load) are not in the feature set.
- Model beats persistence by only 0.06 R² and loses on MAE; predicted range 3.3–10.1 mg/L against an observed 3.0–17.0, so it cannot flag hypoxic events.
- HSI labels remain synthetic; only the falsification test constrains them.
- Neither image has been built. `tests/test_deployment.py` (20 tests) covers what a daemon is not needed for; the base images, Linux wheels, libgomp, the non-root user against a bind mount, nginx's reading of its own config and the HEALTHCHECK loops are unverified. CI's `containers` job is where that gets settled.
- The frontend bakes its API URL in at build time (Vite substitutes `import.meta.env`), so a deployed bundle cannot be repointed without rebuilding.
- No auth, no TLS, no rate limiting. `/explain` costs ~200 ms of SHAP per call and nothing limits it. See `docs/DEPLOYMENT.md`.
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
**Actual**: [Start] → [End]  
**Status**: 🟡 In Progress

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

**Commit**: `git log --oneline | head -1`
```
[Paste commit hash & message]
```

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

**Commit**:
```
[Paste commit hash & message]
```

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

**Date Started**: _______________  
**Date Completed**: _______________  

**Data Statistics**:
```
Dataset: [Name]
Rows: ________
Columns: ________
Date range: ________ to ________
Completeness: ________%
Missing values: ________ (handled via: ________)
Temporal coverage: ________ days
Spatial coverage: ________ river reaches
```

**Features Created**:
```
Hydrodynamic:  [List]
Thermal:       [List]
Water quality: [List]
Sediment:      [List]
Derived:       [List]
```

**Data Quality Issues Found**:
```
[List any outliers, gaps, anomalies discovered]
```

**Correlation Insights**:
```
[Key correlations between environmental variables]
```

**Commit**:
```
[Paste commit hash & message]
```

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
**Actual**: [Start] → [End]  
**Status**: 🟡 In Progress

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

**Date Started**: _______________  
**Date Completed**: _______________  

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
Dataset splits:
  Train: [N] rows (______% of data)
  Val:   [N] rows (______% of data)
  Test:  [N] rows (______% of data)

Training time: ________ seconds
Model size: ________ MB
Training log:
  [Iteration 1]: loss = ____
  [Iteration 50]: loss = ____
  [Final]: loss = ____
```

**Model Performance (Test Set)**:
```
RMSE: ________
MAE:  ________
R²:   ________
Pearson correlation: ________
Nash-Sutcliffe Efficiency (NSE): ________
```

**Commit**:
```
[Paste commit hash & message]
```

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

**Date Started**: _______________  
**Date Completed**: _______________  

**Feature Importance (Top 10)**:
```
1. [Feature]: [Importance score]
2. [Feature]: [Importance score]
3. [Feature]: [Importance score]
...
```

**Critical Thresholds Discovered**:
```
1. [Variable] < [Value] → [Consequence]
   Example: DO < 2 mg/L → Severe oxygen stress

2. [Variable] > [Value] → [Consequence]
   Example: Temperature > 30°C → Heat stress

3. [Var1] × [Var2] interaction:
   Example: Temp > 25°C AND DO < 4 mg/L → Critical combination
```

**Interaction Findings**:
```
Strongest interactions detected:
  - [Feature1] × [Feature2]: strength = ______
  - [Feature1] × [Feature2]: strength = ______
  - [Feature1] × [Feature2]: strength = ______
```

**SHAP Explanation Example**:
```
Prediction: 0.72 (habitat suitability)
Expected value: 0.65
Feature contributions:
  + DO: +0.15 (beneficial)
  - Temperature: -0.08 (harmful)
  + Flow: +0.05 (beneficial)
  - Sediment: -0.04 (harmful)
Base value + sum of contributions = 0.72 ✓

Interpretation: "Moderate habitat suitability despite elevated temperature, 
primarily driven by adequate dissolved oxygen and flow conditions."
```

**Commit**:
```
[Paste commit hash & message]
```

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

**Date Started**: _______________  
**Date Completed**: _______________  

**Validation Results Table**:
```
| Model              | RMSE  | MAE  | R²   | Temporal | Spatial | Event  | Status      |
|--------------------|-------|------|------|----------|---------|--------|-------------|
| XGBoost (ML)       | 0.12  | 0.08 | 0.84 | ✓ PASS   | ✓ PASS  | ✓ PASS | ✅ Selected |
| Random Forest (ML) | 0.15  | 0.10 | 0.81 | ✓ PASS   | ✓ PASS  | ✓ PASS | ✓ Backup    |
| Linear Regression  | 0.25  | 0.18 | 0.62 | ✗ FAIL   | ✗ FAIL  | ✗ FAIL | ✗ Rejected  |
| Hydraulic-only     | 0.30  | 0.22 | 0.55 | ✗ FAIL   | ✗ FAIL  | ✗ FAIL | ✗ Baseline  |
| Persistence        | 0.35  | 0.28 | 0.45 | ✗ FAIL   | ✗ FAIL  | ✗ FAIL | ✗ Baseline  |
```

**Temporal Generalization**:
```
Training period: [Date] to [Date]
Validation period: [Date] to [Date]
Test period: [Date] to [Date]

Model maintains RMSE < 0.15 across all three periods: ✓ GOOD
No overfitting detected: ✓ CONFIRMED
```

**Spatial Generalization**:
```
Trained on reaches: [List]
Tested on reaches: [List]

Performance drop from training to spatial test: _______%
Threshold: acceptable if < 10%
Result: ✓ ACCEPTABLE
```

**Baseline Comparison**:
```
XGBoost vs Hydraulic-only:
  RMSE improvement: ______% better
  R² improvement: ______% better
  Verdict: ML provides significant value

XGBoost vs Linear regression:
  RMSE improvement: ______% better
  Verdict: Non-linear relationships matter
```

**Commit**:
```
[Paste commit hash & message]
```

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
**Actual**: [Start] → [End]  
**Status**: 🟡 In Progress

### 3a. API Design & Core Endpoints
- [x] FastAPI app skeleton
- [ ] Pydantic request/response schemas
- [ ] POST /predict endpoint
- [ ] POST /batch_predict endpoint
- [ ] GET /health endpoint
- [ ] Swagger/OpenAPI docs auto-generated
- [ ] Unit tests for routes

**Date Started**: _______________  
**Date Completed**: _______________  

**API Endpoints Implemented**:
```
✓ POST /predict
  Input: EnvironmentalState (temperature, DO, flow, sediment, depth, velocity)
  Output: PredictionResponse (score 0-1, confidence, message)
  Status: ✓ WORKING

✓ POST /batch_predict
  Input: CSV file with multiple observations
  Output: Array of predictions
  Status: ✓ WORKING

✓ GET /health
  Output: {"status": "ok", "service": "AquaNexus API"}
  Status: ✓ WORKING

Planned:
  - POST /scenario_run
  - GET /explain/{id}
  - POST /interactive_analysis
```

**Example Request/Response**:
```json
// POST /predict
Request:
{
  "timestamp": "2024-01-15T10:30:00",
  "temperature": 24.5,
  "dissolved_oxygen": 7.2,
  "flow": 150.0,
  "suspended_sediment": 45.3,
  "depth": 1.8,
  "velocity": 0.85
}

Response:
{
  "prediction": 0.72,
  "confidence": 0.92,
  "message": "Habitat suitability is moderate (72%). Environmental conditions support limited biological communities.",
  "timestamp": "2024-01-15T10:30:00"
}
```

**API Testing**:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{...}'

Response status: 200 ✓
Response time: ________ ms
```

**Commit**:
```
[Paste commit hash & message]
```

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
**Actual**: [Start] → [End]  
**Status**: 🟡 In Progress

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

**Date Started**: _______________  
**Date Completed**: _______________  

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

**Commit**:
```
[Paste commit hash & message]
```

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

**Date Started**: _______________  
**Date Completed**: _______________  

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
✗ CONTRIBUTING.md      not written — single-author project
```

**Deployment Options**: `docker compose up` brings up both services (API 8000,
dashboard 3000). Cloud targets are deliberately not documented as recipes —
nothing here is production-hardened, and `docs/DEPLOYMENT.md` lists what would
have to change first (no auth, no TLS, no rate limiting, artefacts mounted from
the host).

**Final Commit**:
```
[Paste commit hash & message]
```

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

- [ ] **Phase 1**: Data pipeline complete, 50K+ records processed
- [ ] **Phase 2**: Model trained (R² ≥ 0.80), explainability working
- [ ] **Phase 3**: FastAPI backend fully functional, Docker ready
- [ ] **Phase 4**: React frontend polished, tests passing, CI/CD active
- [ ] **Documentation**: README, ARCHITECTURE, API_REFERENCE, DEPLOYMENT
- [ ] **GitHub**: Clean commit history, professional repo
- [ ] **Model file**: data/models/xgboost_v1.joblib (serialized)
- [ ] **DEVLOG.md**: Complete development chronicle
- [ ] **Production ready**: docker-compose up → works perfectly

---

## Key Metrics & Results

**Data Pipeline**:
```
Dataset: [River name]
Records: ________
Features: ________
Time period: ________ to ________
Completeness: ________%
```

**Model Performance**:
```
RMSE: ________
MAE: ________
R²: ________
Pearson correlation: ________
Temporal generalization: ✓ PASS
Spatial generalization: ✓ PASS
Event-based generalization: ✓ PASS
```

**System Performance**:
```
Prediction latency: ________ ms
Batch prediction (100 rows): ________ ms
API response time: ________ ms
Frontend load time: ________ ms
```

**Code Quality**:
```
Test coverage: ________%
Code style: [pylint/flake8 score]
Documentation: [lines of docs]
Commits: [N] with clear messages
```

---

## Important Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| River | [River name] | [Why this river] |
| Model | XGBoost | Fast, good performance, explainable |
| Explainability | SHAP | Industry standard, interpretable |
| Backend | FastAPI | Async, fast, auto-docs, easy to deploy |
| Frontend | React + TS | Professional, type-safe, large ecosystem |
| Deployment | Docker | Reproducibility, cloud-ready |

---

## Challenges & Solutions

### Challenge 1: [Issue]
**Problem**: _______________
**Solution Attempted**: _______________
**Result**: ✓ RESOLVED / 🟡 PARTIAL / ✗ ONGOING
**Notes**: _______________

### Challenge 2: [Issue]
**Problem**: _______________
**Solution Attempted**: _______________
**Result**: ✓ RESOLVED / 🟡 PARTIAL / ✗ ONGOING
**Notes**: _______________

---

## Lessons Learned

1. [Key learning from development]
2. [What went well]
3. [What could be improved]
4. [For next version]

---

## Next Steps (Beyond PoC)

- [ ] Integrate real biological validation data
- [ ] Deploy to cloud (AWS, Google Cloud, Heroku)
- [ ] Add user authentication & database
- [ ] Expand to multiple Japanese rivers
- [ ] Publish paper/results
- [ ] Community feedback & iterations

---

## Final Notes

**Project Status**: 🟢 **COMPLETE**

This proof-of-concept successfully demonstrates:
- HEC-RAS integration and hydraulic modeling workflow
- Machine learning pipeline for ecosystem prediction
- Full-stack development (Python backend + React frontend)
- Professional software engineering practices (Docker, CI/CD, tests, docs)

**Recommended for**: Portfolio, GitHub showcase, IGES professor discussion, future collaborators

**Last Updated**: _______________  
**Last Commit**: _______________  

---

**Developer Signature**: kiruthick01  
**Date**: _______________

