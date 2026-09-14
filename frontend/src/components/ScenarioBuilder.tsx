/**
 * What-if controls over a baseline state.
 *
 * Changing discharge now carries the hydraulics with it - the API re-interpolates
 * depth, velocity and width from the HEC-RAS sweep - so the state stays
 * physically coherent. What it still is not is a forecast: the sweep is
 * precomputed steady-flow profiles, and below about 2 m³/s the model is
 * unreliable whatever it is fed. Both facts are shown with the answer, not
 * filed under a help link.
 */

import { useState } from "react";
import { useLocation } from "react-router-dom";

import { scenarioUrl, type ScenarioLink } from "../services/permalink";
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
  /** Fractional changes, owned by the page so the URL can carry them. */
  modifications: Record<string, number>;
  onModificationsChange: (modifications: Record<string, number>) => void;
  name: string;
  onNameChange: (name: string) => void;
  /** The whole scenario, for the shareable link. */
  link: ScenarioLink;
  onRun: () => Promise<ScenarioResponse | undefined>;
  busy?: boolean;
}

export default function ScenarioBuilder({
  baseline,
  modifications,
  onModificationsChange,
  name,
  onNameChange,
  link,
  onRun,
  busy,
}: Props) {
  const [result, setResult] = useState<ScenarioResponse | null>(null);

  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    onModificationsChange(preset.modifications);
    onNameChange(preset.name);
  };

  const run = async () => {
    const response = await onRun();
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
                  onModificationsChange({
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
          <ShareLink link={link} />
          <input
            aria-label="Scenario name"
            type="text"
            value={name}
            onChange={(event) => onNameChange(event.target.value)}
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

/** How long to wait for the clipboard before showing the URL instead. */
const CLIPBOARD_TIMEOUT_MS = 1200;

/**
 * Copy a link to this scenario.
 *
 * Two ways the clipboard is not there when you ask for it, both met in a real
 * browser rather than guessed at:
 *
 * - `navigator.clipboard` is undefined on a plain http origin that is not
 *   localhost - which is exactly how this would first be deployed.
 * - `writeText` **never settles** while `document.visibilityState` is
 *   "hidden": Chrome defers the write until the page is visible again, so
 *   awaiting it can hang forever. A rejection would have been fine; silence is
 *   not, because neither the success nor the failure branch ever runs and the
 *   button just sits there.
 *
 * So the wait is bounded, and anything other than a prompt success falls back
 * to showing the URL in a field to select. A share control that quietly does
 * nothing is worse than no share control.
 */
function ShareLink({ link }: { link: ScenarioLink }) {
  // The router's path, not window.location.pathname: they agree in the browser
  // and differ under a MemoryRouter, and the one that is always right is the
  // one the app is actually routing on.
  const { pathname } = useLocation();
  const [copied, setCopied] = useState(false);
  const [shown, setShown] = useState<string | null>(null);

  const share = async () => {
    const url = scenarioUrl(link, window.location.origin, pathname);
    const timeout = new Promise<never>((_, reject) =>
      window.setTimeout(reject, CLIPBOARD_TIMEOUT_MS, new Error("clipboard timeout")),
    );
    try {
      await Promise.race([navigator.clipboard.writeText(url), timeout]);
      setCopied(true);
      setShown(null);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setShown(url);
    }
  };

  return (
    <>
      <button type="button" className="secondary" onClick={share}
              title="Copies the inputs, not the answer: opening the link re-runs
                     the scenario against the model being served then.">
        {copied ? "Link copied" : "Copy link"}
      </button>
      {shown && (
        <input
          aria-label="Scenario link"
          readOnly
          value={shown}
          onFocus={(event) => event.target.select()}
          style={{
            font: "inherit",
            padding: "0.35rem 0.5rem",
            border: "1px solid var(--rule)",
            borderRadius: "6px",
            flex: "1 1 100%",
          }}
        />
      )}
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
        <strong>This is a model sensitivity, not a simulation.</strong> Depth,
        velocity and width were re-interpolated from the HEC-RAS sweep at this
        discharge, so the state is coherent — but the sweep is precomputed
        steady-flow profiles, not a fresh run, and a driver changed on its own
        can still describe conditions this river does not produce.
      </div>
      {Object.keys(result.derived).length > 0 && (
        <p className="small muted">
          Carried with the discharge:{" "}
          {Object.entries(result.derived)
            .map(([name, value]) => `${name} ${value}`)
            .join(", ")}
          .
        </p>
      )}

      <CaveatList caveats={result.caveats} labels={result.labels} />
    </section>
  );
}
