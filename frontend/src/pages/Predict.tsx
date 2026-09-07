/**
 * Predict: one state in, one prediction and its explanation out.
 */

import { useState } from "react";

import EnvironmentalInput, { DEFAULT_STATE } from "../components/EnvironmentalInput";
import ExplainabilityPanel from "../components/ExplainabilityPanel";
import PredictionCard from "../components/PredictionCard";
import { ErrorBanner } from "../components/Provenance";
import TargetPicker from "../components/TargetPicker";
import { api, ApiError } from "../services/api";
import type {
  EnvironmentalState,
  ExplanationResponse,
  PredictionResponse,
  Target,
} from "../types";
import type { ModelState } from "../useModels";

export default function Predict({ models }: { models: ModelState }) {
  const [state, setState] = useState<EnvironmentalState>(DEFAULT_STATE);
  const [target, setTarget] = useState<Target>("dissolved_oxygen");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = async (withExplanation: boolean) => {
    setBusy(true);
    setError(null);
    try {
      const prediction = await api.predict(state, target);
      setResult(prediction);
      if (withExplanation) {
        // The first explanation per model builds a SHAP explainer server-side
        // and takes a couple of seconds; later ones are fast.
        setExplanation(await api.explain(state, target));
      } else {
        setExplanation(null);
      }
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h1>Predict</h1>
      <p className="muted prose">
        Set an environmental state and ask either model what it makes of it. Values
        outside the training range are answered and flagged rather than refused —
        the caller is told where the evidence stops and decides for themselves.
      </p>

      {error && <ErrorBanner message={error} />}

      <div className="split">
        <section className="card" aria-label="Environmental state">
          <div className="card-title">
            <h2>Environmental state</h2>
          </div>
          <TargetPicker
            value={target}
            onChange={setTarget}
            models={models.models}
            disabled={busy}
          />
          <div style={{ height: "0.9rem" }} />
          <EnvironmentalInput
            state={state}
            onChange={setState}
            model={models.byTarget(target)}
            disabled={busy}
          />
          <div className="button-row">
            <button type="button" onClick={() => predict(false)} disabled={busy}>
              {busy ? "Predicting…" : "Predict"}
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => predict(true)}
              disabled={busy}
            >
              Predict and explain
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setState(DEFAULT_STATE);
                setResult(null);
                setExplanation(null);
              }}
              disabled={busy}
            >
              Reset
            </button>
          </div>
        </section>

        <div>
          {result ? (
            <PredictionCard result={result} model={models.byTarget(target)} />
          ) : (
            <section className="card">
              <p className="muted" style={{ margin: 0 }}>
                The state on the left is a warm July day on the Ayase — 24.5 °C
                at 12 m³/s. Change it, then ask a model what it makes of it.
              </p>
            </section>
          )}
          {explanation && <ExplainabilityPanel explanation={explanation} />}
        </div>
      </div>
    </>
  );
}
