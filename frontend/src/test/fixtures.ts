import { vi } from "vitest";
import type { HealthResponse, ModelInfo, PredictionResponse, StationInfo } from "../types";

export const health: HealthResponse = {
  status: "ok",
  version: "0.1.0",
  models_loaded: ["geo", "full"],
  stations: 5883,
  detail: null,
};

const model = (mode: "geo" | "full", r2: number, recall: number): ModelInfo => ({
  mode,
  summary: `${mode} summary`,
  features: ["air_temp", "elevation"],
  low_do_weight: 1,
  trained_on: { rows: 510527, stations: 5295 },
  held_out_strict: { r2, mae: 0.75, recall_low: recall, precision_low: 0.4 },
  held_out_note: "Trained FY<=2019, scored on unseen stations in FY2021-22.",
  importance: { air_temp: 0.5, elevation: 0.2 },
  interval_half_widths: [2.3, 1.7, 1.4],
  interval_bin_edges: [6, 9],
  built: "2026-10-18",
  commit: "abc1234",
});

export const models: ModelInfo[] = [model("full", 0.7154, 0.57), model("geo", 0.4845, 0.2)];

export const station: StationInfo = {
  station_id: 1110100,
  name: "畷橋",
  water_body: "綾瀬川上流",
  pref_code: 11,
  latitude: 35.88,
  longitude: 139.74,
  elevation: 3,
  last_date: "2022-12-15",
  last_do: 10,
  station_do_mean: 7.5,
  station_do_n: 120,
};

export const prediction: PredictionResponse = {
  mode: "geo",
  prediction: 6.11,
  lower: 3.91,
  upper: 8.32,
  level: 0.9,
  interpretation: "6.11 mg/L - adequate for most species",
  missing_features: ["rain_1d"],
  out_of_range: [],
  contributions: { air_temp: -1.2, elevation: 0.3, bias: 7.0 },
  notes: [],
  caveats: ["Trained on monthly river samples."],
};

export function mockFetch(routes: Record<string, unknown>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const key = Object.keys(routes).find((path) => url.includes(path));
    if (!key) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    const body = routes[key];
    const status = body instanceof Error ? 503 : 200;
    return new Response(JSON.stringify(body instanceof Error ? { detail: body.message } : body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fn);
  return calls;
}
