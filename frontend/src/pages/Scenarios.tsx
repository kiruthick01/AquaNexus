/**
 * Scenarios: change a driver, see what the model does with it.
 */

import { useState } from "react";

import EnvironmentalInput, { DEFAULT_STATE } from "../components/EnvironmentalInput";
import { ErrorBanner } from "../components/Provenance";
import ScenarioBuilder from "../components/ScenarioBuilder";
import TargetPicker from "../components/TargetPicker";
import { api, ApiError } from "../services/api";
import type { EnvironmentalState, ScenarioResponse, Target } from "../types";
import type { ModelState } from "../useModels";

export default function Scenarios({ models }: { models: ModelState }) {
  const [baseline, setBaseline] = useState<EnvironmentalState>(DEFAULT_STATE);
  const [target, setTarget] = useState<Target>("dissolved_oxygen");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (
    modifications: Record<string, number>,
    name: string,
  ): Promise<ScenarioResponse | undefined> => {
    setBusy(true);
    setError(null);
    try {
      return await api.scenario(baseline, modifications, target, name);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
      return undefined;
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h1>Scenarios</h1>
      <p className="muted prose">
        Fractional changes against a baseline state. Change the discharge and
        the hydraulics follow, re-interpolated from the HEC-RAS sweep — but the
        sweep is precomputed, so the answer is a sensitivity of the fitted
        relationship rather than a forecast.
      </p>

      {error && <ErrorBanner message={error} />}

      <div className="split">
        <section className="card" aria-label="Baseline state">
          <div className="card-title">
            <h2>Baseline</h2>
          </div>
          <TargetPicker
            value={target}
            onChange={setTarget}
            models={models.models}
            disabled={busy}
          />
          <div style={{ height: "0.9rem" }} />
          <EnvironmentalInput
            state={baseline}
            onChange={setBaseline}
            model={models.byTarget(target)}
            disabled={busy}
          />
        </section>

        <div>
          <ScenarioBuilder baseline={baseline} onRun={run} busy={busy} />
        </div>
      </div>
    </>
  );
}
