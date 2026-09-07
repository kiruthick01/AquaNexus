/**
 * The prediction readout.
 *
 * Layout order is deliberate: provenance badge, then the number, then the
 * interpretation, then anything flagged out of range, then the caveats. A
 * screenshot cropped to the number alone should still carry the badge.
 */

import type { PredictionResponse } from "../types";
import { CaveatList, ProvenanceBadge } from "./Provenance";

const TARGET_NAMES: Record<string, string> = {
  dissolved_oxygen: "Dissolved oxygen",
  hsi: "Habitat suitability",
};

export default function PredictionCard({ result }: { result: PredictionResponse }) {
  const digits = result.target === "hsi" ? 3 : 2;

  return (
    <section className="card" aria-label="Prediction">
      <div className="card-title">
        <h2>{TARGET_NAMES[result.target] ?? result.target}</h2>
        <ProvenanceBadge labels={result.labels} />
      </div>

      <div className="readout">
        <span className="readout-value">{result.prediction.toFixed(digits)}</span>
        <span className="readout-unit">{result.unit}</span>
      </div>
      <p className="interpretation">{result.interpretation}</p>

      <div className="stat-row">
        <div>
          <div className="stat-label">Uncertainty</div>
          <div className="stat-value">
            {result.uncertainty === null ? (
              <span className="muted" title="This model cannot express a spread; a constant would be invented">
                not available
              </span>
            ) : (
              `± ${result.uncertainty.toFixed(digits)} ${result.unit}`
            )}
          </div>
        </div>
        <div>
          <div className="stat-label">Labels</div>
          <div className="stat-value">
            {result.labels === "SYNTHETIC" ? "generated" : "measured"}
          </div>
        </div>
      </div>

      {result.out_of_range.length > 0 && (
        <div className="warning" role="status">
          <strong>Outside the training range.</strong> The model has never seen
          these conditions, and the answer above is extrapolation:
          <ul>
            {result.out_of_range.map((warning) => (
              <li key={warning.feature}>
                <code>{warning.feature}</code> = {warning.value} · trained on{" "}
                {warning.training_min}–{warning.training_max}
              </li>
            ))}
          </ul>
        </div>
      )}

      <CaveatList caveats={result.caveats} labels={result.labels} />
    </section>
  );
}
