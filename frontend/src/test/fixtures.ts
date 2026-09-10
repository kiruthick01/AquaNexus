/**
 * Response fixtures copied from the real API.
 *
 * Taken from a live `uvicorn aquanexus.api.app:app` run against the shipped
 * artefacts, so a schema change on the backend shows up here as a failing test
 * rather than as a blank panel in the browser.
 */

import type {
  ExplanationResponse,
  HealthResponse,
  HoldoutResponse,
  ModelInfo,
  PredictionResponse,
  ScenarioResponse,
} from "../types";

export const health: HealthResponse = {
  status: "ok",
  version: "0.1.0",
  river: "Ayase (綾瀬川)",
  models_loaded: ["dissolved_oxygen", "hsi"],
  detail: null,
};

export const oxygenModel: ModelInfo = {
  target: "dissolved_oxygen",
  unit: "mg/L",
  model_type: "linear",
  labels: "observed",
  n_train: 138,
  features: [
    "water_temp",
    "air_temp",
    "discharge",
    "month_sin",
    "month_cos",
    "reach_depth",
    "reach_velocity",
    "reach_top_width",
    "reach_froude",
    "do_saturation",
  ],
  metrics: { rmse: 1.785, mae: 1.274, r2: 0.394, validation: "grouped CV" },
  caveats: [
    "Labels are real measurements from the 公共用水域 monitoring record.",
    "Barely beats a persistence baseline - 0.009 R2 - and loses to it on MAE.",
  ],
  training_ranges: {
    water_temp: [5.1, 32.5],
    discharge: [0.17, 73.71],
    reach_depth: [2.13, 4.89],
  },
  target_range: [3.0, 17.0],
};

export const hsiModel: ModelInfo = {
  target: "hsi",
  unit: "index 0-1",
  model_type: "xgboost",
  labels: "SYNTHETIC",
  n_train: 4717,
  features: ["depth", "velocity", "water_temp", "dissolved_oxygen"],
  metrics: { rmse: 0.024, r2: 0.993 },
  caveats: [
    "SYNTHETIC LABELS. HSI is generated from ecological response curves in this repository, not measured in the field.",
  ],
  training_ranges: { water_temp: [5.1, 32.5], depth: [0.1, 5.6] },
  target_range: [0.0, 0.989],
};

/**
 * The held-out river, copied from a live `GET /holdout`.
 *
 * The numbers are kept at full precision on purpose: the page rounds them for
 * display, and a fixture pre-rounded to three places would let a formatting bug
 * pass unnoticed.
 */
export const holdout: HoldoutResponse = {
  river: "Naka",
  river_ja: "中川",
  trained_on: "Ayase (綾瀬川)",
  n: 192,
  n_stations: 5,
  generated: "2026-09-10T16:41:19+00:00",
  headline:
    "Inside the ranges it was fitted on, the unchanged Ayase model scores R² +0.336 on a river it has never seen — against +0.394 at home. Outside them it scores -0.903, worse than predicting this river's mean.",
  pooled: [
    {
      label: "Ayase model, unchanged",
      n: 192,
      rmse: 2.1610485737752327,
      mae: 1.559405591508457,
      r2: -0.08064521274104886,
      bias: -1.2869603105070382,
    },
    {
      label: "mean of this river",
      n: 192,
      rmse: 2.078849080203729,
      mae: 1.6684244791666665,
      r2: 0.0,
      bias: 0,
    },
    {
      label: "persistence",
      n: 187,
      rmse: 1.7479858539057305,
      mae: 1.2005347593582887,
      r2: 0.2981190232884331,
      bias: -0.04438502673796792,
    },
    {
      label: "trained on this river",
      n: 192,
      rmse: 1.494709857447184,
      mae: 0.9495220773561494,
      r2: 0.48302698544251976,
      bias: -0.1096,
    },
  ],
  by_evidence: [
    {
      label: "inside Ayase training ranges",
      n: 129,
      rmse: 1.6721,
      mae: 1.0958,
      r2: 0.336338579115393,
      bias: -0.7729,
    },
    {
      label: "outside (extrapolation)",
      n: 63,
      rmse: 2.9169,
      mae: 2.5081,
      r2: -0.9028,
      bias: -2.3402,
    },
  ],
  by_station: [
    {
      sub_reach: "中川上流",
      station: "51道橋",
      n: 36,
      mean_discharge: 3.794166666666667,
      observed_do: 8.466666666666667,
      rmse: 1.1057387150258342,
      r2: 0.5590493707408196,
      bias: -0.511762952854452,
      in_training_range: true,
    },
    {
      sub_reach: "中川中流",
      station: "46八条橋",
      n: 48,
      mean_discharge: 75.0854,
      observed_do: 8.7583,
      rmse: 3.5537,
      r2: -1.0102,
      bias: -3.2472,
      in_training_range: false,
    },
  ],
  home_metrics: {
    rmse: 1.7848778680440163,
    mae: 1.2737725347933389,
    r2: 0.3944476804125533,
    validation: "grouped CV, each station held out",
  },
  document: "docs/HOLDOUT_RIVER.md",
  caveats: [
    "The model is unchanged: loaded from disk as the API serves it, with no refitting or recalibration against the Naka.",
    "Pooled over the river the transfer scores R² -0.081. Read the split, not the pooled figure — it averages two different regimes.",
    "63 of 192 observations fall outside the Ayase training ranges. These are the rows /predict has flagged as out of range since the API was first served; this is what that flag is worth.",
  ],
};

export const prediction: PredictionResponse = {
  target: "dissolved_oxygen",
  prediction: 4.95,
  unit: "mg/L",
  interpretation: "4.95 mg/L - hypoxic stress likely",
  labels: "observed",
  uncertainty: null,
  out_of_range: [],
  contributions: null,
  caveats: oxygenModel.caveats,
};

export const outOfRangePrediction: PredictionResponse = {
  ...prediction,
  prediction: 0,
  interpretation: "0.00 mg/L - acutely lethal for most freshwater fish",
  out_of_range: [
    { feature: "discharge", value: 900, training_min: 0.17, training_max: 73.71 },
  ],
};

export const syntheticPrediction: PredictionResponse = {
  target: "hsi",
  prediction: 0.513,
  unit: "index 0-1",
  interpretation: "Habitat suitability is moderate (0.51)",
  labels: "SYNTHETIC",
  uncertainty: 0.04,
  out_of_range: [],
  contributions: null,
  caveats: hsiModel.caveats,
};

export const explanation: ExplanationResponse = {
  target: "dissolved_oxygen",
  prediction: 6.427,
  baseline: 7.073,
  contributions: [
    { feature: "reach_froude", contribution: -1.054, value: 0.075 },
    { feature: "reach_depth", contribution: 0.849, value: 1.8 },
    { feature: "do_saturation", contribution: -0.731, value: 8.4 },
    { feature: "discharge", contribution: 0.188, value: 12 },
    { feature: "water_temp", contribution: 0.102, value: 24.5 },
  ],
  collinear_pairs: [
    ["water_temp", "do_saturation"],
    ["reach_velocity", "reach_top_width"],
  ],
  caveats: oxygenModel.caveats,
};

export const scenario: ScenarioResponse = {
  scenario_name: "Drought",
  target: "dissolved_oxygen",
  unit: "mg/L",
  labels: "observed",
  baseline_prediction: 6.77,
  scenario_prediction: 6.92,
  change: 0.15,
  percent_change: 2.2,
  applied: { discharge: 4.8 },
  derived: { depth: 2.4148, velocity: 0.3183, top_width: 56.4246 },
  interpretation: "Drought improves the outcome by 0.150 mg/L.",
  out_of_range: [],
  caveats: oxygenModel.caveats,
};
