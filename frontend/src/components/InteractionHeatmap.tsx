/**
 * Response surface over two drivers, computed with one /batch_predict call.
 *
 * This is the honest version of the "interaction heatmap" the plan asked for.
 * The Phase 2b interaction estimate was retracted - it flipped sign between
 * stations - so the grid is presented as what the *model* does across a plane,
 * with the caveat that most of that plane is a combination the river never
 * produces: discharge and velocity are coupled through the hydraulic model, and
 * holding one fixed while sweeping the other describes an impossible state.
 */

import type { EnvironmentalState, Target } from "../types";

export interface GridResult {
  xValues: number[];
  yValues: number[];
  predictions: number[];
  unit: string;
  outOfRangeCount: number;
}

export const GRID_STEPS = 9;

export function linspace(low: number, high: number, steps: number): number[] {
  if (steps < 2) return [low];
  const step = (high - low) / (steps - 1);
  return Array.from({ length: steps }, (_, index) => low + index * step);
}

/** Row-major states for a temperature (y) by discharge (x) grid. */
export function buildGrid(
  base: EnvironmentalState,
  xValues: number[],
  yValues: number[],
): EnvironmentalState[] {
  const states: EnvironmentalState[] = [];
  for (const temperature of yValues) {
    for (const discharge of xValues) {
      states.push({ ...base, water_temp: temperature, discharge });
    }
  }
  return states;
}

/** Blue (low) through pale (mid) to red (high), matching the project figures. */
export function colourFor(value: number, low: number, high: number): string {
  if (!Number.isFinite(value) || high <= low) return "#e9edf1";
  const t = Math.min(1, Math.max(0, (value - low) / (high - low)));
  const stops: [number, number, number][] = [
    [192, 57, 43],
    [230, 126, 34],
    [242, 232, 207],
    [46, 134, 171],
  ];
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const [r1, g1, b1] = stops[index];
  const [r2, g2, b2] = stops[index + 1];
  const mix = (a: number, b: number) => Math.round(a + (b - a) * fraction);
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
}

const CELL = 46;
const PAD_LEFT = 68;
const PAD_BOTTOM = 46;
const PAD_TOP = 10;

export default function InteractionHeatmap({
  grid,
  target,
}: {
  grid: GridResult;
  target: Target;
}) {
  const { xValues, yValues, predictions } = grid;
  const finite = predictions.filter(Number.isFinite);
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const digits = target === "hsi" ? 2 : 1;

  const width = PAD_LEFT + xValues.length * CELL + 12;
  const height = PAD_TOP + yValues.length * CELL + PAD_BOTTOM;

  return (
    <figure>
      <div className="scroll-x">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          role="img"
          aria-label={`Predicted ${target} across water temperature and discharge`}
          style={{ maxWidth: "100%" }}
        >
          {yValues.map((temperature, row) =>
            xValues.map((discharge, column) => {
              const value = predictions[row * xValues.length + column];
              const x = PAD_LEFT + column * CELL;
              // Rows are drawn bottom-up so temperature increases upward.
              const y = PAD_TOP + (yValues.length - 1 - row) * CELL;
              return (
                <g key={`${row}-${column}`}>
                  <title>
                    {`${temperature.toFixed(1)} °C, ${discharge.toFixed(1)} m³/s → ` +
                      `${Number.isFinite(value) ? value.toFixed(digits) : "n/a"} ${grid.unit}`}
                  </title>
                  <rect
                    x={x}
                    y={y}
                    width={CELL - 2}
                    height={CELL - 2}
                    rx="2"
                    fill={colourFor(value, low, high)}
                  />
                  <text
                    x={x + (CELL - 2) / 2}
                    y={y + (CELL - 2) / 2 + 4}
                    textAnchor="middle"
                    fontSize="10.5"
                    fill="#12202f"
                  >
                    {Number.isFinite(value) ? value.toFixed(digits) : "—"}
                  </text>
                </g>
              );
            }),
          )}

          {yValues.map((temperature, row) => (
            <text
              key={`y-${row}`}
              x={PAD_LEFT - 8}
              y={PAD_TOP + (yValues.length - 1 - row) * CELL + CELL / 2}
              textAnchor="end"
              fontSize="10.5"
              fill="var(--ink-2)"
            >
              {temperature.toFixed(0)}°
            </text>
          ))}
          {xValues.map((discharge, column) => (
            <text
              key={`x-${column}`}
              x={PAD_LEFT + column * CELL + (CELL - 2) / 2}
              y={height - PAD_BOTTOM + 22}
              textAnchor="middle"
              fontSize="10.5"
              fill="var(--ink-2)"
            >
              {discharge < 10 ? discharge.toFixed(1) : discharge.toFixed(0)}
            </text>
          ))}
          <text
            x={PAD_LEFT + (xValues.length * CELL) / 2}
            y={height - 8}
            textAnchor="middle"
            fontSize="11"
            fill="var(--ink)"
          >
            discharge (m³/s)
          </text>
          <text
            x={-(PAD_TOP + (yValues.length * CELL) / 2)}
            y={15}
            transform="rotate(-90)"
            textAnchor="middle"
            fontSize="11"
            fill="var(--ink)"
          >
            water temperature (°C)
          </text>
        </svg>
      </div>

      <div className="legend">
        <span>
          <i className="swatch" style={{ background: colourFor(low, low, high) }} />
          {low.toFixed(digits)} {grid.unit}
        </span>
        <span>
          <i className="swatch" style={{ background: colourFor(high, low, high) }} />
          {high.toFixed(digits)} {grid.unit}
        </span>
        {grid.outOfRangeCount > 0 && (
          <span style={{ color: "var(--flag)" }}>
            {grid.outOfRangeCount} of {predictions.length} cells fall outside the
            training range
          </span>
        )}
      </div>

      <figcaption>
        Each cell is one model call. Temperature and discharge are swept
        independently, so much of this plane is a state the river never produces —
        discharge and velocity are coupled through the hydraulic model, and only
        discharge moves here.
      </figcaption>
    </figure>
  );
}
