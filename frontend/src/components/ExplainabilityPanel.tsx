/**
 * SHAP contributions as a diverging bar chart, in inline SVG.
 *
 * The collinearity warning is rendered *above* the chart, not below it. Ten of
 * the dissolved-oxygen model's feature pairs correlate above 0.9 - the
 * hydraulic features are all derived from discharge through the same HEC-RAS
 * model - so SHAP divides credit between them arbitrarily and the bar order is
 * not a ranking. A reader who sees the chart first has already drawn the wrong
 * conclusion.
 */

import type { ExplanationResponse } from "../types";
import { CaveatList } from "./Provenance";

const ROW_HEIGHT = 26;
const LABEL_WIDTH = 148;
const CHART_WIDTH = 460;

export default function ExplainabilityPanel({
  explanation,
}: {
  explanation: ExplanationResponse;
}) {
  const rows = explanation.contributions.filter((c) => Math.abs(c.contribution) > 1e-9);
  const largest = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);
  const height = Math.max(rows.length * ROW_HEIGHT + 28, 60);
  const midpoint = LABEL_WIDTH + (CHART_WIDTH - LABEL_WIDTH) / 2;
  const halfSpan = (CHART_WIDTH - LABEL_WIDTH) / 2 - 46;

  const collinear = new Set(explanation.collinear_pairs.flat());
  const unit = explanation.target === "hsi" ? "" : " mg/L";

  return (
    <section className="card" aria-label="Explanation">
      <div className="card-title">
        <h2>Why this number</h2>
        <span className="small muted">
          {explanation.baseline.toFixed(2)} baseline, {explanation.prediction.toFixed(2)}{" "}
          here
        </span>
      </div>

      {explanation.collinear_pairs.length > 0 && (
        <div className="warning" role="note">
          <strong>
            {explanation.collinear_pairs.length} feature pairs move together
          </strong>{" "}
          (correlated above 0.9 in training). SHAP splits credit between such
          features arbitrarily, so read them as one combined effect rather than a
          ranking. Marked <span aria-hidden="true">◈</span> below.
          <div className="small muted" style={{ marginTop: "0.4rem" }}>
            {explanation.collinear_pairs.map((pair) => (
              <span key={pair.join()} className="pair">
                {pair.join(" with ")}
              </span>
            ))}
          </div>
        </div>
      )}

      {rows.length === 0 ? (
        <p className="muted">No feature moved this prediction.</p>
      ) : (
        <figure>
          <svg
            viewBox={`0 0 ${CHART_WIDTH} ${height}`}
            width="100%"
            height={height}
            role="img"
            aria-label="Per-feature SHAP contributions"
          >
            <line
              x1={midpoint}
              y1={4}
              x2={midpoint}
              y2={height - 20}
              stroke="var(--rule)"
            />
            {rows.map((row, index) => {
              const y = index * ROW_HEIGHT + 6;
              const width = (Math.abs(row.contribution) / largest) * halfSpan;
              const positive = row.contribution > 0;
              return (
                <g key={row.feature}>
                  <text
                    x={LABEL_WIDTH - 8}
                    y={y + 13}
                    textAnchor="end"
                    fontSize="11"
                    fill="var(--ink)"
                  >
                    {collinear.has(row.feature) ? "◈ " : ""}
                    {row.feature}
                  </text>
                  <rect
                    x={positive ? midpoint : midpoint - width}
                    y={y + 3}
                    width={width}
                    height={ROW_HEIGHT - 12}
                    rx="2"
                    fill={positive ? "var(--gauge)" : "var(--flag)"}
                  />
                  <text
                    x={positive ? midpoint + width + 5 : midpoint - width - 5}
                    y={y + 13}
                    textAnchor={positive ? "start" : "end"}
                    fontSize="10.5"
                    fill="var(--ink-2)"
                  >
                    {row.contribution > 0 ? "+" : ""}
                    {row.contribution.toFixed(2)}
                  </text>
                </g>
              );
            })}
            <text
              x={midpoint}
              y={height - 6}
              textAnchor="middle"
              fontSize="10"
              fill="var(--ink-2)"
            >
              contribution to the prediction{unit}
            </text>
          </svg>
          <div className="legend">
            <span>
              <i className="swatch" style={{ background: "var(--gauge)" }} />
              raises the prediction
            </span>
            <span>
              <i className="swatch" style={{ background: "var(--flag)" }} />
              lowers it
            </span>
            <span>◈ shares credit with a correlated feature</span>
          </div>
          <figcaption>
            Contributions sum to prediction − baseline ={" "}
            {(explanation.prediction - explanation.baseline).toFixed(3)}. SHAP
            describes what the model does, not what the river does.
          </figcaption>
        </figure>
      )}

      <CaveatList caveats={explanation.caveats} />
    </section>
  );
}
