/**
 * Types mirroring `aquanexus.api.schemas`.
 *
 * `labels` is the field that matters most here: one of the two served models is
 * trained on labels this project generated, and every component that renders a
 * number is expected to render its provenance beside it.
 */

export type Target = "dissolved_oxygen" | "hsi";
export type Labels = "observed" | "SYNTHETIC";

export interface EnvironmentalState {
  water_temp?: number;
  air_temp?: number;
  dissolved_oxygen?: number;
  discharge?: number;
  depth?: number;
  velocity?: number;
  suspended_solids?: number;
  ph?: number;
  nitrogen_total?: number;
  phosphorus_total?: number;
  month?: number;
}

export interface RangeWarning {
  feature: string;
  value: number;
  training_min: number;
  training_max: number;
}

export interface PredictionResponse {
  target: Target;
  prediction: number;
  unit: string;
  interpretation: string;
  labels: Labels;
  uncertainty: number | null;
  out_of_range: RangeWarning[];
  contributions: Record<string, number> | null;
  caveats: string[];
}

export interface BatchPredictionResponse {
  target: Target;
  unit: string;
  labels: Labels;
  n: number;
  predictions: number[];
  out_of_range_count: number;
  caveats: string[];
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
  value: number | null;
}

export interface ExplanationResponse {
  target: Target;
  prediction: number;
  baseline: number;
  contributions: FeatureContribution[];
  collinear_pairs: string[][];
  caveats: string[];
}

export interface ScenarioResponse {
  scenario_name: string;
  target: Target;
  unit: string;
  labels: Labels;
  baseline_prediction: number;
  scenario_prediction: number;
  change: number;
  percent_change: number | null;
  applied: Record<string, number>;
  interpretation: string;
  out_of_range: RangeWarning[];
  caveats: string[];
}

export interface ModelInfo {
  target: Target;
  unit: string;
  model_type: string;
  labels: Labels;
  n_train: number;
  features: string[];
  metrics: Record<string, number | string>;
  caveats: string[];
  /** Observed [min, max] per feature, so a client can show where evidence ends. */
  training_ranges: Record<string, number[]>;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  river: string;
  models_loaded: string[];
  detail: string | null;
}
