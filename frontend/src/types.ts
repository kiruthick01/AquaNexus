/** Shapes returned by the AquaNexus API. They mirror src/aquanexus/api/schemas.py. */

export type Mode = "geo" | "nowcast" | "full" | "full_alert";

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  models_loaded: string[];
  stations: number;
  detail: string | null;
}

export interface HeldOut {
  n?: number;
  r2?: number;
  mae?: number;
  rmse?: number;
  mae_low?: number;
  recall_low?: number;
  precision_low?: number;
}

export interface ModelInfo {
  mode: Mode;
  summary: string;
  features: string[];
  low_do_weight: number;
  trained_on: { rows?: number; stations?: number; source?: string; rounds?: number };
  held_out_strict: HeldOut;
  held_out_classifier: Partial<{
    auc: number;
    avg_precision: number;
    recall_at_p50: number;
    base_rate: number;
    brier: number;
    ece: number;
  }>;
  held_out_note: string;
  importance: Record<string, number>;
  interval_half_widths: number[];
  interval_bin_edges: number[];
  built: string;
  commit: string;
}

export interface StationInfo {
  station_id: number;
  name: string | null;
  water_body: string | null;
  pref_code: number;
  latitude: number;
  longitude: number;
  elevation: number | null;
  last_date: string;
  last_do: number;
  station_do_mean: number;
  station_do_n: number;
}

export type Weather = Partial<{
  air_temp: number;
  air_temp_3d: number;
  air_temp_7d: number;
  air_temp_30d: number;
  rain_1d: number;
  rain_3d: number;
  rain_7d: number;
  rain_14d: number;
  rain_30d: number;
  days_since_rain: number;
  solar: number;
  wind: number;
}>;

export interface PredictionRequest {
  date: string;
  mode: Mode;
  station_id?: number;
  location?: { latitude?: number; longitude?: number };
  weather?: Weather;
  history?: { prev_do?: number };
  neighbors?: { nb_do_now?: number };
  explain?: boolean;
}

export interface RangeWarning {
  feature: string;
  value: number;
  training_min: number;
  training_max: number;
}

export interface PredictionResponse {
  mode: Mode;
  prediction: number;
  lower: number;
  upper: number;
  probability_below_5: number | null;
  level: number;
  interpretation: string;
  missing_features: string[];
  out_of_range: RangeWarning[];
  contributions: Record<string, number> | null;
  notes: string[];
  caveats: string[];
}
