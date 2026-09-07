/**
 * Typed client for the AquaNexus API.
 *
 * `fetch` rather than axios (which AQUANEXUS-PLAN.md §4a names): the client is
 * a hundred lines of request building, the platform has had `fetch` for years,
 * and a dependency here would need mocking in every component test.
 *
 * Errors carry the API's own `detail` string. The backend answers 503 with a
 * reason when a model or its explainer is unavailable, and that reason is far
 * more useful to a user than "request failed".
 */

import type {
  BatchPredictionResponse,
  EnvironmentalState,
  ExplanationResponse,
  HealthResponse,
  ModelInfo,
  PredictionResponse,
  ScenarioResponse,
  Target,
} from "../types";

export const API_BASE: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // A dead backend is the most likely failure in local use, and the browser's
    // own message ("Failed to fetch") does not say which service is missing.
    throw new ApiError(0, `cannot reach the API at ${API_BASE}`);
  }

  if (!response.ok) {
    throw new ApiError(response.status, await errorDetail(response));
  }
  return (await response.json()) as T;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors arrive as a list of per-field objects.
    if (Array.isArray(detail)) {
      return detail
        .map((item) => `${item?.loc?.slice(1).join(".") ?? "request"}: ${item?.msg}`)
        .join("; ");
    }
    return `request failed with ${response.status}`;
  } catch {
    return `request failed with ${response.status}`;
  }
}

const post = <T>(path: string, payload: unknown): Promise<T> =>
  request<T>(path, { method: "POST", body: JSON.stringify(payload) });

export const api = {
  health: () => request<HealthResponse>("/health"),

  models: () => request<ModelInfo[]>("/models"),

  predict: (state: EnvironmentalState, target: Target, explain = false) =>
    post<PredictionResponse>("/predict", { state, target, explain }),

  batchPredict: (states: EnvironmentalState[], target: Target) =>
    post<BatchPredictionResponse>("/batch_predict", { states, target }),

  explain: (state: EnvironmentalState, target: Target) =>
    post<ExplanationResponse>("/explain", { state, target }),

  scenario: (
    baseline: EnvironmentalState,
    modifications: Record<string, number>,
    target: Target,
    scenarioName: string,
  ) =>
    post<ScenarioResponse>("/scenario_run", {
      scenario_name: scenarioName,
      baseline,
      modifications,
      target,
    }),
};
