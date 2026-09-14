/**
 * Scenarios: change a driver, see what the model does with it.
 *
 * The scenario lives in the URL. What is shared is the question - target,
 * baseline, changes - and never the answer, so opening a link re-runs it
 * against the model being served now, with that model's caveats attached. See
 * `services/permalink.ts`.
 */

import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import EnvironmentalInput, { DEFAULT_STATE } from "../components/EnvironmentalInput";
import { ErrorBanner } from "../components/Provenance";
import ScenarioBuilder from "../components/ScenarioBuilder";
import TargetPicker from "../components/TargetPicker";
import { api, ApiError } from "../services/api";
import { decodeScenario, encodeScenario, type ScenarioLink } from "../services/permalink";
import type { EnvironmentalState, ScenarioResponse, Target } from "../types";
import type { ModelState } from "../useModels";

const FALLBACK: ScenarioLink = {
  target: "dissolved_oxygen",
  baseline: DEFAULT_STATE,
  modifications: { discharge: -0.6 },
  name: "Drought",
};

export default function Scenarios({ models }: { models: ModelState }) {
  const [searchParams, setSearchParams] = useSearchParams();
  // Read once: after this the controls own the state, and re-reading on every
  // render would fight the user for the sliders.
  const [initial] = useState<ScenarioLink>(() =>
    decodeScenario(searchParams, FALLBACK),
  );

  const [baseline, setBaseline] = useState<EnvironmentalState>(initial.baseline);
  const [target, setTarget] = useState<Target>(initial.target);
  const [modifications, setModifications] = useState<Record<string, number>>(
    initial.modifications,
  );
  const [name, setName] = useState(initial.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const link: ScenarioLink = { target, baseline, modifications, name };

  const run = async (): Promise<ScenarioResponse | undefined> => {
    setBusy(true);
    setError(null);
    // Replace rather than push: a slider explored ten times should not cost the
    // reader ten presses of the back button to leave the page.
    setSearchParams(encodeScenario(link), { replace: true });
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
          <ScenarioBuilder
            baseline={baseline}
            modifications={modifications}
            onModificationsChange={setModifications}
            name={name}
            onNameChange={setName}
            link={link}
            onRun={run}
            busy={busy}
          />
        </div>
      </div>
    </>
  );
}
