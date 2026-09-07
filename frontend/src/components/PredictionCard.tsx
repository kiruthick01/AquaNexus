/**
 * The reading, drawn as a gauge staff.
 *
 * A number alone says nothing about whether it is high or low for this river.
 * The band is the range actually measured on the Ayase; the needle is this
 * prediction. A value near the edge of the evidence looks near the edge, and a
 * value outside it is drawn outside the band rather than described as outside
 * in a sentence somebody may not read.
 *
 * Order is deliberate: provenance, then the number, then where it sits, then
 * what it means, then anything flagged, then the caveats hung beside it. A
 * screenshot cropped to the number still carries its provenance.
 */

import type { ModelInfo, PredictionResponse } from "../types";
import { CaveatList, ProvenanceBadge } from "./Provenance";

const TARGET_NAMES: Record<string, string> = {
  dissolved_oxygen: "Dissolved oxygen",
  hsi: "Habitat suitability",
};

export function GaugeStaff({
  value,
  range,
  unit,
  digits,
}: {
  value: number;
  range: [number, number];
  unit: string;
  digits: number;
}) {
  const [low, high] = range;
  const span = high - low || 1;
  const fraction = (value - low) / span;
  const outside = value < low || value > high;
  const position = Math.min(1, Math.max(0, fraction));

  return (
    <div className="gauge">
      <div className="gauge-track">
        <div className="gauge-band" />
        <div
          className={`gauge-needle ${outside ? "out" : ""}`}
          style={{ left: `${position * 100}%` }}
        />
      </div>
      <div className="gauge-scale">
        <span>{low.toFixed(digits)}</span>
        <span>{high.toFixed(digits)}</span>
      </div>
      <p className="gauge-caption">
        {outside
          ? `Outside everything measured here — the record runs ${low.toFixed(digits)} to ${high.toFixed(digits)} ${unit}.`
          : `Measured range on this river: ${low.toFixed(digits)}–${high.toFixed(digits)} ${unit}.`}
      </p>
    </div>
  );
}

export default function PredictionCard({
  result,
  model,
}: {
  result: PredictionResponse;
  model?: ModelInfo;
}) {
  const digits = result.target === "hsi" ? 3 : 2;
  const range = model?.target_range;
  const hasRange = range !== undefined && range.length === 2;

  return (
    <section className="card" aria-label="Prediction">
      <div className="card-title">
        <h2>{TARGET_NAMES[result.target] ?? result.target}</h2>
        <ProvenanceBadge labels={result.labels} />
      </div>

      <div className="reading">
        <span className="reading-value">{result.prediction.toFixed(digits)}</span>
        <span className="reading-unit">{result.unit}</span>
      </div>

      {hasRange && (
        <GaugeStaff
          value={result.prediction}
          range={[range[0], range[1]]}
          unit={result.unit}
          digits={digits}
        />
      )}

      <p className="interpretation">{result.interpretation}</p>

      <div className="facts">
        <div>
          <div className="stat-label">Uncertainty</div>
          <div className="stat-value">
            {result.uncertainty === null ? (
              <span
                className="muted"
                title="This model cannot express a spread, and a constant would be invented"
              >
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
          <strong>Outside the training range.</strong> The model has not seen
          these conditions, so the reading above is extrapolation:
          <ul>
            {result.out_of_range.map((warning) => (
              <li key={warning.feature}>
                <span className="mono">{warning.feature}</span> {warning.value} ·
                measured {warning.training_min}–{warning.training_max}
              </li>
            ))}
          </ul>
        </div>
      )}

      <CaveatList caveats={result.caveats} labels={result.labels} />
    </section>
  );
}
