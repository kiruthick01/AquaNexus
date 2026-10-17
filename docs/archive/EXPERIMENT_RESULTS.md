# Experiment Results — Consolidated Summary

Human-readable companion to `experiments/results.csv`. Full narrative for
each row is in `docs/EXPERIMENTS.md`; this page is the cross-phase view and
the answers to the questions `docs/ML_ROADMAP.md` set out to answer.

All results below use the canonical dissolved-oxygen dataset
(`data/processed/ayase_do_dataset.csv`, 138 real observations, 4 Ayase
stations) unless a row says otherwise, and the canonical
leave-one-station-out protocol (`docs/VALIDATION.md`) unless noted. The
canonical point-prediction baseline throughout is Ridge: **RMSE 1.785, R²
0.394** (`docs/ML_METHODOLOGY.md`), which itself barely beats a persistence
baseline (RMSE 1.818, R² 0.385) — every number below should be read against
that context, not in isolation.

## Results table

| Phase | Method | Status | Key result |
|---|---|---|---|
| 1. Forecasting | lag/lead/walk-forward infra | **Blocked** | 0/138 rows usable at any horizon (t+1/t+3/t+7) |
| 2. Uncertainty | Bootstrap 90% PI | Viable | Coverage 0.920, width 6.74 mg/L, RMSE 1.794 |
| 2. Uncertainty | Quantile regression 90% PI | Viable, unreliable | Coverage 0.703 (undercovers) |
| 3. Conformal | Split conformal, in-domain | Viable | Coverage 0.857 (n=28) |
| 3. Conformal | Split conformal, station-held-out | Viable | Coverage 0.870 pooled; worst station 0.583 |
| 3. Conformal | Split conformal, cross-river | Viable | Coverage 0.849 pooled; out-of-range station 0.479 |
| 4. Remote sensing | NDVI/NDWI/MNDWI infra | **Blocked** | 0/9 stations have a coordinate |
| 5. Deep learning | MLP (321 params) | Viable, negative result | RMSE 2.358, R² **-0.057** (loses to the mean) |
| 5. Deep learning | LSTM | **Blocked** | Same blocker as Phase 1 |
| 6. Transfer | Zero-shot (existing) | Viable | Pooled R² -0.081 (full 192-row Naka) |
| 6. Transfer | Domain-aligned | Viable | R² -0.089 on held-out subset (small real gain) |
| 6. Transfer | Fine-tuned, best shrinkage | Viable | R² **0.372** — best of any Phase 6 method |
| 6. Transfer | Fine-tuned, weak shrinkage | Viable, negative result | R² **-6.548** — worst of any method tested |
| 6. Transfer | Frozen-head (MLP) | Viable | R² 0.349 |
| 7. Bayesian | Pooled linear | Viable | Coverage 0.913, RMSE 2.612 (underperforms Ridge) |
| 7. Bayesian | Hierarchical | Viable | Coverage 0.971, RMSE 2.741 |
| 8. GNN | GCN/GAT infra | **Blocked** | 0/2 rivers have a recorded edge list |

Full per-row detail: `experiments/results.csv`. Existing ablations
(hydraulic-only vs. full feature set, `ML_METHODOLOGY.md` "Does the hydraulic
model actually help?") predate this roadmap and were preserved, not rerun.

## Answers to the project's closing questions

**1. Does temporal information improve DO prediction?** Untested, not
negative — the real record (irregular ~monthly grab sampling) cannot support
a forecasting evaluation at all (0/138 usable rows). The infrastructure is
ready; the data is not.

**2. Does spatial information improve generalization?** The station-held-out
protocol (already the project's default) is what every result in this table
is evaluated under — spatial generalization is not an add-on here, it is the
baseline discipline. Explicit spatial *features* (coordinates) do not exist
to test separately (Phase 4's blocker).

**3. Do remote-sensing features add predictive value?** Untested — 0/9
stations have a coordinate, so no feature was ever computed to test.

**4. Does deep learning outperform classical ML given the available
dataset?** No. The MLP's R² (-0.057) is below the constant-mean floor
(0.000), a clear, measured loss, not a marginal one.

**5. How reliable are the model's uncertainty estimates?** Mixed and
now precisely characterized: bootstrap is close to nominal (0.920 vs 0.90
target); quantile regression badly undercovers (0.703); split conformal
looks stable pooled (~0.85-0.87 across three domain-shift regimes) but hides
a real per-station failure (as low as 0.479); both Bayesian models are
calibrated-to-conservative (0.913, 0.971) but wide.

**6. Does conformal prediction achieve its intended coverage?** Close,
pooled, in every regime tested — but a pooled number is not sufficient to
certify it: one station or river subset can miss the 0.90 target badly
(0.479-0.583) while the pooled figure still looks fine.

**7. Does uncertainty degrade under domain shift?** Yes, but not where a
pooled number would show it. Split conformal's pooled coverage barely moved
from in-domain to cross-river (0.857 → 0.849); the degradation is
concentrated entirely in the specific subgroup already known to be outside
the model's training range (`docs/HOLDOUT_RIVER.md`), not spread evenly.

**8. Does transfer learning improve cross-river performance?** Yes, when
adequately regularized — the best fine-tuning result (R² 0.372) and
frozen-head adaptation (R² 0.349) both clearly beat zero-shot (R² -0.081 to
-0.133). Under-regularized fine-tuning made things dramatically worse (R²
-6.548), the single worst result measured anywhere in this project. The
technique's name did not determine the outcome; its regularization strength
relative to the tiny (29-row) adaptation slice did.

**9. Does Bayesian modeling provide useful calibrated uncertainty?** Yes for
calibration (both models meet or exceed the 0.90 target), no for point
accuracy or sharpness — traced to a generic weakly-informative prior
regularizing less than Ridge's tuned penalty, not a modeling error.

**10. Does explicit river-network structure improve predictions through
GNNs?** Untested — no real edge list or sufficient node count exists for
either river.

**11. Which features actually matter?** Unchanged by this roadmap:
`ML_METHODOLOGY.md`'s SHAP analysis and hydraulic-only ablation remain the
project's answer (thermal/seasonal drivers dominate; hydraulics contribute
less than the original spec assumed). No new capability in this roadmap
altered that ranking, since none produced a competitive alternative model to
re-derive it from.

**12. Where does the model fail?** Consistently, across every phase that
touched it: **55畷橋** (an Ayase station) and, on the Naka, **46八条橋** (the
one out-of-training-range station) are where every uncertainty method's
coverage and every point model's accuracy is worst. This is now a
cross-validated finding, not a single phase's artifact — it recurs in
Phase 2, Phase 3, and implicitly in Phase 6/7's held-out evaluations.

**13. Does added complexity produce scientifically meaningful improvement?**
No, and often the opposite. The MLP lost to the mean. The Bayesian models
lost to Ridge on accuracy. Under-regularized fine-tuning lost badly to doing
nothing. The one place complexity helped (adequately-regularized fine-tuning,
frozen-head adaptation) helped because it was *specifically regularized* for
the tiny data available, not because it was more sophisticated. The
consistent finding across all eight phases is that **regularization strength
relative to sample size, not model sophistication, is what separates a
working method from a failing one on this dataset.**
