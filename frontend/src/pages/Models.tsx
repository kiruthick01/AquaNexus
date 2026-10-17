/**
 * Models: each served model, what it needs, and its held-out accuracy.
 */

import type { ModelInfo } from "../types";
import type { ModelState } from "../useModels";

const fmt = (value: number | undefined, digits = 2) =>
  value === undefined ? "–" : value.toFixed(digits);

function TopFeatures({ model }: { model: ModelInfo }) {
  const top = Object.entries(model.importance).slice(0, 6);
  const max = Math.max(...top.map(([, v]) => v), 1e-9);
  return (
    <ul className="small" style={{ listStyle: "none", padding: 0, margin: 0 }}>
      {top.map(([name, value]) => (
        <li key={name} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <span className="mono" style={{ width: "9rem" }}>
            {name}
          </span>
          <span
            aria-hidden="true"
            style={{
              display: "inline-block",
              height: "0.6rem",
              width: `${(value / max) * 8}rem`,
              background: "var(--gauge)",
            }}
          />
          <span className="mono">{value.toFixed(2)}</span>
        </li>
      ))}
    </ul>
  );
}

export default function Models({ models }: { models: ModelState }) {
  return (
    <>
      <h1>Models</h1>
      <p className="muted prose">
        Accuracy is measured on stations the model never saw, in fiscal years 2021-22 after training
        on years up to 2019. The served models are retrained on all years, so these numbers describe
        the recipe, not a score on the served file itself.
      </p>
      {models.models.map((model) => (
        <section className="card" key={model.mode} aria-label={model.mode}>
          <div className="card-title">
            <h2 className="mono">{model.mode}</h2>
          </div>
          <p>{model.summary}</p>
          <div className="facts">
            <div>
              <div className="stat-label">R²</div>
              <div className="stat-value">{fmt(model.held_out_strict.r2)}</div>
            </div>
            <div>
              <div className="stat-label">Mean error</div>
              <div className="stat-value">{fmt(model.held_out_strict.mae)} mg/L</div>
            </div>
            <div>
              <div className="stat-label">Below 5 mg/L caught</div>
              <div className="stat-value">{fmt((model.held_out_strict.recall_low ?? 0) * 100, 0)}%</div>
            </div>
            <div>
              <div className="stat-label">Alarms that are right</div>
              <div className="stat-value">
                {fmt((model.held_out_strict.precision_low ?? 0) * 100, 0)}%
              </div>
            </div>
          </div>
          {model.held_out_classifier.auc !== undefined && (
            <p className="small">
              Chance of DO below 5 mg/L: ranking AUC {fmt(model.held_out_classifier.auc)}, average
              precision {fmt(model.held_out_classifier.avg_precision)} against a base rate of{" "}
              {fmt((model.held_out_classifier.base_rate ?? 0) * 100, 1)}%; calibration error{" "}
              {fmt(model.held_out_classifier.ece, 3)}.
            </p>
          )}
          <h3>What it relies on (mean absolute contribution, mg/L)</h3>
          <TopFeatures model={model} />
          <p className="small muted">
            {model.held_out_note} Trained on {model.trained_on.rows?.toLocaleString() ?? "?"} rows,
            90% interval half-width {model.interval_half_widths.map((w) => w.toFixed(1)).join(" / ")}{" "}
            mg/L by predicted-DO band.
          </p>
        </section>
      ))}
    </>
  );
}
