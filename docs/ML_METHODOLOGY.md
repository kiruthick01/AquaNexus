# ML Methodology

Working document. Supersedes parts of `../ML_STRATEGY.md`, which was written before
the data audit and assumed a 4-year hourly series on the Yodo.

## Standing caveat

HSI labels are generated, not observed. Any metric reported here describes how well
the model recovers a known function of its own inputs. See the README's
"Scope and honest caveats".

The observed monitoring record does give one genuine check on the label function —
see [Falsification test](#falsification-test-for-the-synthetic-labels) below.

---

## Corrections to ML_STRATEGY.md §3

Implemented in `aquanexus.data.preprocessor`. Three deliberate departures:

### 1. Lagged and rolling features are not produced

§3.2 specifies `*_lag1/6/24` and `*_ma7/ma30` features. Those assume a continuous
hourly chronology. The dataset is a designed experiment over flow space, where rows
are independent samples and "the previous hour" does not exist. Generating them
would manufacture autocorrelation that the physics never produced.

### 2. The thermal stress formula contradicted its own description

§3.2 gives:

```
thermal_stress_index = 1 / (1 + exp(-a * (temp - optimal_temp)))
  Penalty for temp > 25°C or < 10°C
```

That expression is a monotonic sigmoid — it rises with temperature and **never
penalises cold water**, directly contradicting the prose beneath it. The prose is
ecologically correct, so the implementation uses a two-sided Gaussian response about
the optimum (17.5 °C), symmetric in cold and warm departures.

### 3. The oxygen stress formula was unbounded

§3.2 gives:

```
oxygen_stress_index = exp(-b * (optimal_do - do))
```

When DO exceeds the optimum, `(optimal_do - do)` is negative and the expression grows
without limit — it cannot be a stress index. Replaced with a bounded response: 1 at
or below the critical threshold (2 mg/L), 0 at or above ample (8 mg/L), squared in
between because the ecological cost of losing oxygen accelerates as levels fall.

### Also: DO saturation is computed, not assumed

§3.1 treats `optimal_do` as a constant 10 mg/L. Saturation is strongly temperature
dependent — 14.6 mg/L at 0 °C against 8.3 at 25 °C — so a fixed reference would
misread ordinary summer warming as oxygen depletion. `do_saturation()` implements the
Benson–Krause equation (APHA 4500-O) with an elevation pressure correction, and is
tested against the published saturation table to ±0.03 mg/L.

---

## Observed design space (Ayase, FY2022–24, n=216)

Derived with `observed_ranges()` from the real monitoring record. These bounds define
the simulation sweep, replacing the guessed ranges in §3.1:

| Variable | q0.01 | q0.99 | ML_STRATEGY §3.1 said | 
|---|---|---|---|
| water_temp (°C) | 5.6 | 32.5 | 5–35 ✓ |
| discharge (m³/s) | 0.25 | 64.2 | **50–500 ✗** |
| dissolved_oxygen (mg/L) | 3.4 | 12.9 | 0–14 ✓ |
| bod (mg/L) | 0.8 | 8.7 | 0–10 ✓ |
| nitrogen_total (mg/L) | 1.7 | 6.2 | **0–5 ✗** |
| phosphorus_total (mg/L) | 0.14 | 0.38 | 0–1 ✓ |
| suspended_solids (mg/L) | 4.0 | 41.9 | **0–500 ✗** |

The discharge range is the significant one: the plan's lower bound of 50 m³/s is
above the Ayase's 99th percentile. Simulating that range would have produced a
dataset with almost no overlap with the real river.

Quantiles rather than min/max, so one fouled sample cannot stretch the design space.

---

## Falsification test for the synthetic labels

The observed record carries a real, independent relationship:

```
DO above 25 °C:  5.13 mg/L  (n=59)
DO below 15 °C:  8.74 mg/L  (n=72)
```

Worst observed condition: **31.2 °C at 3.1 mg/L DO — 42% saturation**
(57綾瀬川合流点前, 2024-09-18). Best: 17.8 °C at 84% saturation, zero computed stress.

**If the generated HSI does not degrade across that gradient, the label function is
wrong.** This does not make the labels real, but it makes them refutable, which is a
materially stronger position than the original plan had.

Note the record also contains genuine **supersaturation** — DO up to 17.0 mg/L, and
106% saturation on 2023-04-13. This is real (spring algal photosynthesis), not error.
`do_deficit` is deliberately allowed to go negative rather than being clipped, since
supersaturation is itself a eutrophication signal.

---

## Synthetic labels — corrections to §4

Implemented in `aquanexus.data.synthetic`.

### Habitat profile: the spec named the wrong fish

§4.2 specifies coldwater fish (trout, char) — which do not live in the Ayase, a
lowland Kanto river reaching 32.5 °C. Under trout optima every summer sample scores
essentially zero, destroying the signal. `LOWLAND_WARMWATER` (eurythermal cyprinids
— オイカワ, コイ, フナ) is the default; `COLDWATER` retains the spec's parameters
for comparison.

### Four structural errors in §4.3

1. **DO used a two-sided Gaussian**, penalising high oxygen as hard as low. At
   `optimal=10, std=2`, a real observed 17.0 mg/L scores 0.002 — rated as harmful as
   3.0 mg/L. Replaced with a plateau, plus a separate penalty only for extreme
   supersaturation (which genuinely causes gas-bubble trauma).
2. **The interaction penalty's branches were mis-ordered.** With the spec's
   `if/elif`, 26 °C at 3.5 mg/L takes the first branch (−0.15) while the *milder*
   23 °C at 3.5 mg/L takes the second (−0.20) — worse conditions cost less. Replaced
   with a smooth monotonic term.
3. **Sediment thresholds were set for a sediment-laden river.** Only 0.5% of Ayase
   samples exceed the spec's 50 mg/L trigger, making the term nearly inert.
4. **The arithmetic mean let good factors offset lethal ones.** At 33 °C with
   2.5 mg/L DO, optimal depth and velocity scoring 1.0 each pulled HSI to 0.58 —
   "moderate" habitat for unbreathable water. Switched to the **geometric mean**
   (standard HSI practice) plus an explicit **acute-survival multiplier**, since
   below ~1.5–3 mg/L fish asphyxiate regardless of everything else. Lethal
   thresholds are profile parameters: cyprinids tolerate oxygen that kills salmonids.

## Falsification result

Run against 216 real Ayase observations, restricted to a 15–30 °C band:

```
lowland_warmwater      n_in_band=124
  HSI  DO<5   0.617 (n=30)   <   DO>=8   0.781 (n=13)     ✓
  HSI  stressed 0.563 (n=28) <   benign  0.830 (n=18)     ✓
  spearman(HSI, DO) = 0.761                            => PASS
```

**Temperature banding is essential, and getting it wrong inverted the verdict.** The
first version compared DO groups across all samples and reported DO<5 at 0.585
*above* DO≥8 at 0.505 — apparently backwards. The cause was confounding, not a broken
label: dissolved oxygen anti-correlates with temperature, so the most oxygen-rich
samples are winter water, which scores poorly for a warmwater guild. Comparing like
with like reverses it. The Spearman coefficient is reported as a diagnostic only, for
the same reason.

Resulting seasonal curve on real data — spring peak, summer depression from combined
heat and hypoxia, winter depression from cold:

| Month | 1 | 4 | 5 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|
| water temp (°C) | 8.4 | 19.6 | 19.5 | 31.1 | 28.8 | 12.3 |
| DO (mg/L) | 9.0 | 8.5 | 6.8 | 5.3 | 4.7 | 7.1 |
| **HSI** | **0.39** | **0.79** | **0.85** | **0.62** | **0.64** | **0.54** |

Label distribution: mean 0.648, sd 0.199 — close to the mean 0.628 / sd 0.185 that
ML_STRATEGY §4.5 anticipated.

This does not make the labels real. It makes them refutable, and they currently
survive refutation.

## Validation splits — revised

§5's temporal split does not apply; there is no chronology to split on.

- **Held-out flow space** (primary) — train on the interior of the design grid, test
  on an unseen corner such as high discharge with high temperature.
- **Held-out reach** (secondary) — **done, 2026-09-08.** See
  `HOLDOUT_RIVER.md`: the Naka was built through an identical pipeline and
  handed to the unchanged Ayase model. It scores R² +0.336 where the inputs lie
  inside the training ranges, against +0.394 at home, and −0.903 outside them,
  pooling to −0.081. The model's domain is its training range, not "rivers".
- **Held-out extremes** (tertiary) — reframed from "event-based": the extreme
  conditions within the design grid.

## Phase 2a result — and why the numbers are too good

The dataset (`aquanexus.data.dataset`) joins simulated hydraulics to observed
chemistry through the discharge measured alongside each water sample: 7,314 rows,
49 cross-sections × 138 observations, 26 features (four sections excluded, see below).

Benchmark on a **grouped** split (whole observations held out):

| model | RMSE | MAE | R² | skill vs mean |
|---|---|---|---|---|
| random_forest | 0.024 | 0.015 | **0.994** | 0.994 |
| xgboost | 0.024 | 0.015 | **0.993** | 0.993 |
| linear (Ridge) | 0.104 | 0.077 | 0.877 | 0.878 |
| mean (floor) | 0.298 | 0.256 | −0.007 | 0.000 |

**Do not read 0.99 as success.** ML_STRATEGY §9.1 expects R² 0.80–0.88 and §9.3
expects linear regression around 0.55. Both are badly exceeded, and the reason is
methodological, not a modelling triumph:

**The labels are noiseless.** HSI is a deterministic function of the features,
computed by code in this repository. There is no measurement error, no
unexplained ecological variance, nothing stochastic. The model is recovering a
smooth analytic function from its own inputs, which is close to the easiest
possible regression problem. The plan's 0.80–0.88 target implicitly assumed data
with real noise in it.

Probing how easy, with a depth-4 decision tree on single features:

```
depth alone            R² 0.655
velocity alone         R² 0.284
depth + velocity       R² 0.883
dissolved oxygen alone R² 0.067
water temperature      R² 0.032
```

Two consequences worth stating plainly:

1. **The label is dominated by hydraulics, not water quality.** Depth, velocity
   and shear stress carry 89% of feature importance; dissolved oxygen 4%. Within
   one observation the HSI standard deviation is 0.244, against 0.134 between
   observation means — geometry varies more than chemistry does. That is an
   artefact of the join: 49 sections span depths of 0.1–5.6 m at one discharge,
   while chemistry is held constant along the reach.
2. **The depth response is tight relative to the reach.** With an optimum of
   1.0 m and a tolerance of 0.8 m, the deeper sections score near zero, pulling
   the mean HSI down to 0.32 from the 0.65 the chemistry-only labels gave.

**What the benchmark does establish:** the pipeline is correct end to end, the
splits hold groups apart, the model set trains and is comparable, and the metrics
are computed against an explicit floor. What it does **not** establish is
ecological skill, and no amount of tuning here would change that.

**Where the R² gap between splits is informative:** grouped 0.993, spatial 0.994,
flow-extrapolation 0.973, ungrouped-random 0.998. The ungrouped split is highest,
as expected — it straddles groups and rewards memorising near-duplicate rows. The
flow split is lowest, which is the right ordering: extrapolating to unseen
discharges is genuinely harder.

## The real ML target: observed dissolved oxygen

The synthetic-HSI benchmark above measures function recovery. This one does not:
the label is **measured dissolved oxygen** from the monitoring record — a real
number with real instrument and sampling error.

`build_water_quality_dataset()` produces one row per observation (n=138 on the
Ayase, 4 stations with both discharge and DO). Reach-averaged hydraulics at the
observed discharge enter as features, which is how the hydraulic model earns its
place: DO is governed partly by reaeration, and reaeration depends on depth and
velocity.

BOD, nutrients and suspended solids are **deliberately excluded**. They come from
the same bottle as the target, so including them predicts one measurement from
another rather than from the river's physical state, and the hydraulics would
stop mattering.

### Result — grouped cross-validation, each station held out

| model | RMSE (mg/L) | MAE | R² | skill |
|---|---|---|---|---|
| **Ridge (linear)** | **1.785** | 1.274 | **0.394** | 0.394 |
| random forest | 1.884 | 1.425 | 0.325 | 0.325 |
| xgboost | 1.909 | 1.408 | 0.307 | 0.307 |
| DO saturation (physics) | 2.900 | 2.577 | −0.598 | −0.598 |
| mean (floor) | 2.294 | 1.779 | 0.000 | 0.000 |

Three things worth stating:

**Linear beats both tree models.** With 138 rows and 10 features, gradient
boosting overfits and cross-validates worse than a regularised linear fit. The
plan assumes throughout that XGBoost wins; on this dataset it does not, and
reporting it the other way round would be dishonest.

**The saturation baseline fails informatively.** Predicting DO as saturation at
the observed temperature scores R² −0.60 — worse than the mean — yet correlates
at Pearson 0.697. It has the right *shape* and the wrong *level*: bias +2.39 mg/L.
**The Ayase runs a persistent oxygen deficit of roughly 2.4 mg/L below
saturation.** That is a real, measured property of this river, and it is exactly
what a habitat model should care about.

### Does the hydraulic model actually help? (ablation)

Ridge, same grouped CV:

| feature set | RMSE | R² | ΔR² |
|---|---|---|---|
| temperature only | 1.880 | 0.328 | — |
| + season | 1.902 | 0.312 | −0.016 |
| + discharge | 1.929 | 0.293 | −0.019 |
| **+ hydraulics (HEC-RAS)** | **1.902** | **0.312** | **+0.019** |
| hydraulics only | 2.629 | −0.313 | — |

**Modest, but real, and the shape of it is the interesting part.** Adding raw
discharge *hurts* (−0.019). Adding the hydraulic model's transformation of that
same discharge into depth, velocity, width and Froude number *helps* (+0.062 over
discharge, +0.027 over temperature alone).

So the physics-informed transformation carries information the raw driver does
not — which is the project's central claim, demonstrated on real labels. It is a
small effect and should be reported as one; hydraulics alone predict nothing
(R² −0.26), because DO is thermally driven first.

## Phase 2b — explainability (ML_STRATEGY §7.2)

Implemented in `aquanexus.ml.explainer`. Two deviations from the spec, both forced:

**§7.2 Step 2 specifies `TreeExplainer`** because it assumes XGBoost is the chosen
model. On the DO target Ridge cross-validates better, and `TreeExplainer` cannot
explain a Ridge pipeline. The explainer dispatches on model type instead.

**§7.2 Steps 2–3 specify 8,760 background and test records.** The real datasets are
138 rows (DO) and 7,314 (HSI). Background comes from the training partition,
k-means summarised when large.

### Step 6: interaction analysis — and what it cannot answer

Of the pairs tested, **only one is identifiable**:

| pair | correlation | result |
|---|---|---|
| water_temp × discharge | +0.13 | interaction +0.23 mg/L — **not established**, see below |
| water_temp × do_saturation | −0.99 | not identifiable (empty cells) |
| discharge × reach_velocity | +0.89 | not identifiable (empty cells) |

#### Correction (2026-09-07): the synergy was a single-station result

This section previously reported water_temp × discharge as **synergistic at
−1.02 mg/L**, with a correlation of +0.24 — warm water plus high flow depressing
oxygen more than the parts added. Rebuilding the analysis for
`notebooks/04_explainability.ipynb` showed that figure came from explaining a
48-row subset, which is one station (52内匠橋). Over all 138 observations the same
call returns **+0.23 mg/L**, and per station the sign flips:

| subset | n | correlation | interaction |
|---|---|---|---|
| 52内匠橋 | 48 | +0.24 | −0.93 |
| 54槐戸橋 | 36 | +0.48 | −0.53 |
| 55畷橋 | 36 | +0.69 | **+1.72** |
| 57綾瀬川合流点前 | 18 | +0.07 | +0.87 |
| all stations | 138 | +0.13 | +0.23 |

With 18–48 observations per station and a temperature–discharge correlation that
is itself unstable, this design cannot resolve an interaction of that size. The
mechanism remains plausible — storm load arriving with warm water is real in urban
lowland rivers — but plausible is not measured, and it should not have been
reported as a finding. The collinearity count below moves with the explained
subset for the same reason: **ten** pairs at |r| ≥ 0.9 over all 138 rows, twelve
over that one station's 48.

The other pairs cannot be answered from this data at all. When two features
correlate at 0.97+, one corner of the two-by-two design is empty — "high discharge,
low velocity" never occurs — so the interaction is unidentifiable. The explainer
now returns `identifiable: False` with the reason rather than a bare NaN, because
a NaN there reads as "no interaction" when the truth is "cannot tell".

### Collinearity: why individual SHAP ranks here are not trustworthy

**Ten feature pairs correlate at |r| ≥ 0.9** across all 138 observations (twelve
over the 48-row subset the first run used). The hydraulic features are all
derived from discharge through the same model, so depth, velocity, top width and
Froude number are near-duplicates of one another; water temperature and DO
saturation are deterministically related at r = −0.99.

SHAP still sums correctly to each prediction, but **how it divides that total
between collinear features is arbitrary**. The consequence is visible in the
threshold sweep: `water_temp` shows a marginal response span of only 0.32 mg/L and
is flagged not influential, while `do_saturation` — a deterministic function of
water temperature — spans 5.42 mg/L. The model routes the temperature signal
through one of the pair, and reading either in isolation understates it.

`HabitatExplainer.collinearity()` lists these pairs, and `feature_importance()`
documents the caveat. Individual hydraulic feature ranks should be read as one
combined contribution, not a league table.

### Step 5: threshold discovery

Marginal response with all other features at their median, ranked by how far the
prediction moves across each feature's observed range:

| feature | response span (mg/L) | steepest at | influential |
|---|---|---|---|
| do_saturation | 5.42 | 12.08 mg/L | yes |
| reach_top_width | 3.62 | 48.9 m | yes |
| air_temp | 2.33 | 35.2 °C | yes |
| reach_froude | 1.90 | 0.03 | yes |
| reach_depth | 1.70 | 3.41 m | yes |
| discharge | 1.53 | 16.6 m³/s | yes |
| water_temp | 0.32 | 32.4 °C | **no** — see collinearity above |

These are partial-dependence curves: they describe what the model does, not what
the river does. Holding correlated features at their median produces combinations
that may never occur.

## The four flagged cross-sections, and what removing them cost

`scripts/constriction_impact.py`, full results in `CONSTRICTION_IMPACT.md`.

Four sections — RS 12500, 14000, 18000, 24000 — were cut where the centreline
wandered off the channel, giving channels of 11–37 m against neighbours of
34–155 m. They were flagged from Phase 1c and kept, because dropping them was a
judgement nobody had measured.

Measured, it turned out to matter: excluding them moves predictions by 1.10 mg/L
at worst, 64% of the model's error. **They are excluded now, and every number in
this document is computed on the remaining 49.**

The correction made the model look *worse*, on every axis:

| | 53 sections (flawed) | 49 sections (shipped) |
|---|---|---|
| RMSE | 1.713 | **1.785** |
| R² | 0.442 | **0.394** |
| margin over persistence | +0.057 R² | **+0.009 R²** |
| low-flow bias | −2.011 mg/L | **−2.256 mg/L** |
| predicted range | 3.3–10.1 | **4.2–10.2** |
| hydraulic gain over raw discharge | +0.062 R² | **+0.019 R²** |

That direction is the point. The four bad sections were adding structure the
model could fit, and removing them took away a third of the apparent value of
the entire hydraulic pipeline. A metric that improves when you fix your data is
pleasant; one that degrades is informative, and choosing geometry by which
version scores better would have been the actual error.

## Manning's n — how much the uncalibrated roughness is worth

`scripts/manning_sensitivity.py`, full results in `MANNING_SENSITIVITY.md`.

The channel roughness was never calibrated: no gauged rating curve exists for
this reach, so 0.035 is a textbook value for a lowland earth channel. Re-running
the same 12-discharge sweep across 0.025–0.050 puts a number on that choice.

| n | reach depth at 8.1 m³/s | mean shift in predicted DO | worst shift |
|---|---|---|---|
| 0.025 | 2.569 m | 0.717 mg/L | 1.098 mg/L |
| 0.030 | 2.591 m | 0.558 | 0.837 |
| **0.035 (shipped)** | **2.650 m** | — | — |
| 0.040 | 2.670 m | 0.127 | 0.284 |
| 0.045 | 2.689 m | 0.215 | 0.445 |
| 0.050 | 2.748 m | 0.521 | 0.812 |

**Up to 15% of the model's 1.785 mg/L RMSE comes from a constant nobody
measured** — 0.26 mg/L at worst. Depth moves 6.4% across the whole range while
the prediction moves 3.7%, so the model *damps* the hydraulic uncertainty rather
than amplifying it: it leans on temperature far more heavily than on the
channel. The same weakness that makes the hydraulic pipeline add so little also
protects the answer from the roughness being wrong.

Run against the earlier 53-section geometry this study read 64%. Almost all of
that was the four bad cross-sections, not the roughness — which is why it was
re-run after they were removed, and why a sensitivity study is only as current
as the artefacts underneath it.

This does not calibrate anything. It ranks the problem: one gauged
stage-discharge record for this reach would replace the range with a value, and
is worth more than any modelling change currently available.

## Phase 2c — validation and baselines

`aquanexus.ml.validator`. All figures grouped-CV, each station held out.

| model | RMSE | MAE | R² | skill |
|---|---|---|---|---|
| **linear (Ridge)** | **1.785** | 1.274 | **0.394** | 0.394 |
| **persistence** | 1.818 | **1.213** | 0.385 | 0.385 |
| random_forest | 1.884 | 1.425 | 0.325 | 0.325 |
| xgboost | 1.909 | 1.408 | 0.307 | 0.307 |
| mean (floor) | 2.294 | 1.779 | 0.000 | 0.000 |
| hydraulic-only | 2.572 | 1.870 | −0.258 | −0.258 |

### The result that should temper everything else

**The model beats persistence by 0.009 R² — and loses to it on MAE.**

"Same dissolved oxygen as last month at this station" scores R² 0.385 against the
model's 0.394, with a *lower* median error (1.213 vs 1.274 mg/L). For a slowly
varying quantity sampled monthly that is a strong baseline, and it is the one the
spec's comparison table would have omitted.

The model is not useless — it works at sites and times with no previous sample,
which persistence cannot do at all, and it generalises to a station never seen.
But the honest headline is *"marginally better than assuming no change"*, not
*"R² 0.44"*.

**Hydraulics alone score −0.157**, worse than the mean. Consistent with the
earlier ablation: dissolved oxygen is thermally driven first, and hydraulics
modulate rather than determine it.

### Spatial validation — each station held out

| station | n | observed mean | RMSE | bias |
|---|---|---|---|---|
| 55畷橋 | 36 | 8.36 | 2.206 | −1.540 |
| 54槐戸橋 | 36 | 7.39 | 1.793 | −0.444 |
| 57綾瀬川合流点前 | 18 | 6.27 | 1.439 | +1.205 |
| 52内匠橋 | 48 | 6.15 | 1.266 | +0.115 |

The model regresses toward the reach mean: it under-predicts the most oxygenated
station by 1.5 mg/L and over-predicts the least by 1.2. With four stations it
cannot learn site-specific behaviour it has never seen.

### Event-based validation — tails of discharge

| band | n | mean discharge | RMSE | bias |
|---|---|---|---|---|
| low tail | 21 | 1.1 m³/s | **2.532** | **−2.011** |
| middle | 96 | 13.2 | 1.634 | −0.043 |
| high tail | 21 | 51.8 | 0.818 | +0.103 |

**At low flow the model under-predicts oxygen by 2 mg/L** and its error is triple
the high-flow case. That is the worst possible place for this weakness: drought
is when oxygen stress actually threatens habitat, so the model is least reliable
exactly where it would be consulted. Likely because low-flow rows are scarce and
the hydraulic interpolation is clamped at the bottom of the swept range.

Reported here rather than buried: any operational use should treat low-flow
predictions as unreliable.

## Benchmarks — to re-derive

The R² 0.80–0.88 in §9 was set against the hourly-series design and should not be
carried over unexamined. With synthetic labels these numbers measure function
recovery, not ecological skill, and the figure should be reported that way.
