/**
 * Model metadata, fetched once and shared.
 *
 * `/models` carries the held-out accuracy and feature importance every page needs, so it is loaded at the top of the app rather than per page. A
 * degraded backend - one that started without artefacts - returns 503 here, and
 * the message it gives is the one worth showing the user.
 */

import { useCallback, useEffect, useState } from "react";

import { api, ApiError } from "./services/api";
import type { HealthResponse, Mode, ModelInfo } from "./types";

export interface ModelState {
  models: ModelInfo[];
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  byMode: (mode: Mode) => ModelInfo | undefined;
  reload: () => void;
}

export function useModels(): ModelState {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const status = await api.health();
        if (cancelled) return;
        setHealth(status);

        if (status.status !== "ok") {
          setModels([]);
          setError(status.detail ?? "the API started without models");
          return;
        }
        const loaded = await api.models();
        if (!cancelled) setModels(loaded);
      } catch (caught) {
        if (!cancelled) {
          setError(
            caught instanceof ApiError ? caught.message : "could not reach the API",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const byMode = useCallback(
    (mode: Mode) => models.find((model) => model.mode === mode),
    [models],
  );

  return {
    models,
    health,
    loading,
    error,
    byMode,
    reload: () => setAttempt((n) => n + 1),
  };
}
