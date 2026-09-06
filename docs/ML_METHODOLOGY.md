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
- **Held-out reach** (secondary) — still valid. Naka is reserved for this; it has
  contiguous point-cloud coverage and its own monitoring stations.
- **Held-out extremes** (tertiary) — reframed from "event-based": the extreme
  conditions within the design grid.

## Phase 2a result — and why the numbers are too good

The dataset (`aquanexus.data.dataset`) joins simulated hydraulics to observed
chemistry through the discharge measured alongside each water sample: 7,314 rows,
53 cross-sections × 138 observations, 26 features.

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
   artefact of the join: 53 sections span depths of 0.1–5.6 m at one discharge,
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

## Benchmarks — to re-derive

The R² 0.80–0.88 in §9 was set against the hourly-series design and should not be
carried over unexamined. With synthetic labels these numbers measure function
recovery, not ecological skill, and the figure should be reported that way.
