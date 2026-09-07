/**
 * What-if controls over a baseline state.
 *
 * The endpoint moves the model's inputs; it does not re-run HEC-RAS. A drought
 * scenario therefore leaves depth, velocity and width where the baseline put
 * them - a state the river cannot physically be in - and the result is
 * unreliable below about 2 m³/s regardless. Both facts are shown with the
 * answer, not filed under a help link.
 */

import { useState } from "react";

import type { EnvironmentalState, ScenarioResponse } from "../types";
import { CaveatList, ProvenanceBadge } from "./Provenance";

const PRESETS: { name: string; label: string; modifications: Record<string, number> }[] = [
  { name: "Drought", label: "−60% discharge", modifications: { discharge: -0.6 } },
  { name: "Flood", label: "+150% discharge", modifications: { discharge: 1.5 } },
  { name: "Heatwave", label: "+15% water temperature", modifications: { water_temp: 0.15 } },
  {
    name: "Warm low flow",
    label: "+15% temperature, −40% discharge",
    modifications: { water_temp: 0.15, discharge: -0.4 },
  },
];

const ADJUSTABLE: { key: keyof EnvironmentalState; label: string }[] = [
  { key: "discharge", label: "Discharge" },
  { key: "water_temp", label: "Water temperature" },
  { key: "depth", label: "Depth" },
  { key: "velocity", label: "Velocity" },
];

interface Props {
  baseline: EnvironmentalState;
  onRun: (
    modifications: Record<string, number>,
    name: string,
  ) => Promise<ScenarioResponse | undefined>;
  busy?: boolean;
}

export default function ScenarioBuilder({ baseline, onRun, busy }: Props) {
  const [modifications, setModifications] = useState<Record<string, number>>({
    discharge: -0.6,
  });
  const [name, setName] = useState("Drought");
  const [result, setResult] = useState<ScenarioResponse | null>(null);

  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    setModifications(preset.modifications);
    setName(preset.name);
  };

  const run = async () => {
    const response = await onRun(modifications, name);
    if (response) setResult(response);
  };

  const lowFlow =
    (baseline.discharge ?? 0) * (1 + (modifications.discharge ?? 0)) < 2;

  return (
    <>
      <section className="card" aria-label="Scenario builder">
        <div className="card-title">
          <h2>What if</h2>
        </div>

        <div className="button-row" style={{ marginBottom: "0.9rem" }}>
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              className="secondary"
              onClick={() => applyPreset(preset)}
              title={preset.label}
            >
              {preset.name}
            </button>
          ))}
        </div>

        {ADJUSTABLE.map((field) => {
          const fraction = modifications[field.key] ?? 0;
          const current = baseline[field.key];
          const next = current === undefined ? undefined : current * (1 + fraction);
          return (
            <div className="field" key={field.key}>
              <div className="field-head">
                <label htmlFor={`scenario-${field.key}`}>{field.label}</label>
                <span className="field-value">
                  {fraction > 0 ? "+" : ""}
                  {Math.round(fraction * 100)}%
                  {next !== undefined && ` → ${next.toFixed(2)}`}
                </span>
              </div>
              <input
                id={`scenario-${field.key}`}
                type="range"
                min={-1}
                max={2}
                step={0.05}
                value={fraction}
                disabled={busy}
                aria-label={`${field.label} change`}
                onChange={(event) =>
                  setModifications({
                    ...modifications,
                    [field.key]: Number(event.target.value),
                  })
                }
              />
            </div>
          );
        })}

        <div className="button-row">
          <button type="button" onClick={run} disabled={busy}>
            {busy ? "Running…" : "Run scenario"}
          </button>
          <input
            aria-label="Scenario name"
            type="text"
            value={name}
            onChange={(event) => setName(event.target.value)}
            style={{
              font: "inherit",
              padding: "0.35rem 0.5rem",
              border: "1px solid var(--rule)",
              borderRadius: "6px",
              flex: "1 1 10rem",
            }}
          />
        </div>

        {lowFlow && (
          <div className="warning" role="status">
            <strong>Below about 2 m³/s the model is not trustworthy.</strong> It
            is biased roughly −2 mg/L in that band, and most of its low-flow
            evidence comes from a single shallow upstream station. Drought is
            exactly when oxygen matters, and exactly where this model is weakest.
          </div>
        )}
      </section>

      {result && <ScenarioResult result={result} />}
    </>
  );
}

function ScenarioResult({ result }: { result: ScenarioResponse }) {
  const digits = result.target === "hsi" ? 3 : 2;
  const better = result.change > 0;

  return (
    <section className="card" aria-label="Scenario result">
      <div className="card-title">
        <h2>{result.scenario_name}</h2>
        <ProvenanceBadge labels={result.labels} />
      </div>

      <div className="facts" style={{ borderTop: 0, marginTop: 0, paddingTop: 0 }}>
        <div>
          <div className="stat-label">Baseline</div>
          <div className="stat-value">
            {result.baseline_prediction.toFixed(digits)} {result.unit}
          </div>
        </div>
        <div>
          <div className="stat-label">Scenario</div>
          <div className="stat-value">
            {result.scenario_prediction.toFixed(digits)} {result.unit}
          </div>
        </div>
        <div>
          <div className="stat-label">Change</div>
          <div
            className="stat-value"
            style={{ color: better ? "var(--growth)" : "var(--flag)" }}
          >
            {result.change > 0 ? "+" : ""}
            {result.change.toFixed(digits)} {result.unit}
            {result.percent_change !== null &&
              ` (${result.percent_change > 0 ? "+" : ""}${result.percent_change.toFixed(1)}%)`}
          </div>
        </div>
      </div>

      <p className="interpretation">{result.interpretation}</p>

      <div className="warning">
        <strong>This is a model sensitivity, not a simulation.</strong> Changing
        discharge here moves the model's input; it does not re-run HEC-RAS, so
        depth, velocity and width stay at their baseline values — a combination
        the river cannot actually be in.
      </div>

      <CaveatList caveats={result.caveats} labels={result.labels} />
    </section>
  );
}
