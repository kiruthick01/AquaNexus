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
  metrics: { rmse: 1.713, mae: 1.232, r2: 0.442, validation: "grouped CV" },
  caveats: [
    "Labels are real measurements from the 公共用水域 monitoring record.",
    "Beats a persistence baseline by only 0.06 R2 and loses to it on MAE.",
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
  interpretation: "Drought improves the outcome by 0.150 mg/L.",
  out_of_range: [],
  caveats: oxygenModel.caveats,
};
