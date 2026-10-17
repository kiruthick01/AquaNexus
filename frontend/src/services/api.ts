/**
 * Typed client for the AquaNexus API.
 *
 * `fetch` rather than axios: the client is a hundred lines of request building and
 * a dependency here would need mocking in every component test.
 *
 * Errors carry the API's own `detail` string. The backend answers 503 with a
 * reason when a model is unavailable, and that reason is far more useful to a user
 * than "request failed".
 */

import type {
  HealthResponse,
  ModelInfo,
  PredictionRequest,
  PredictionResponse,
  StationInfo,
} from "../types";

declare global {
  interface Window {
    __AQUANEXUS_CONFIG__?: { apiBaseUrl?: string };
  }
}

/**
 * Where the API lives, resolved at page load rather than at build time.
 *
 * Order matters. `VITE_API_BASE_URL` is the local override and wins, because
 * `.env.local` is how you point a dev server at something else. Otherwise the
 * value comes from `/config.js`, which a container rewrites at start-up — that
 * is what lets one built image serve any environment. The final fallback is the
 * default dev backend.
 */
function resolveApiBase(): string {
  const built = import.meta.env.VITE_API_BASE_URL;
  if (typeof built === "string" && built.length > 0) return built;

  const runtime =
    typeof window !== "undefined" ? window.__AQUANEXUS_CONFIG__?.apiBaseUrl : undefined;
  if (typeof runtime === "string" && runtime.length > 0) return runtime;

  return "http://localhost:8000";
}

export const API_BASE: string = resolveApiBase();

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

  stations: (query: string, limit = 12) =>
    request<StationInfo[]>(`/stations?q=${encodeURIComponent(query)}&limit=${limit}`),

  predict: (payload: PredictionRequest) => post<PredictionResponse>("/predict", payload),
};
