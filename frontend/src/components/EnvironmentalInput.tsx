/**
 * The environmental state form.
 *
 * Sliders span the API's physical limits; the readout turns red past the
 * *training* range, which comes from `/models` rather than being hard-coded
 * here - the model is retrained regularly and a copied number would drift.
 *
 * Values outside the training range are allowed. The API answers such requests
 * and flags them, and the form follows the same rule: preventing the question
 * would hide that the model has an edge at all.
 */

import type { EnvironmentalState, ModelInfo } from "../types";

interface FieldSpec {
  key: keyof EnvironmentalState;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
}

/** Slider bounds, matching the physical limits in `aquanexus.api.schemas`. */
export const FIELDS: FieldSpec[] = [
  { key: "water_temp", label: "Water temperature", unit: "°C", min: -2, max: 45, step: 0.1 },
  { key: "air_temp", label: "Air temperature", unit: "°C", min: -30, max: 45, step: 0.1 },
  { key: "discharge", label: "Discharge", unit: "m³/s", min: 0, max: 120, step: 0.1 },
  { key: "depth", label: "Depth", unit: "m", min: 0, max: 10, step: 0.05 },
  { key: "velocity", label: "Velocity", unit: "m/s", min: 0, max: 3, step: 0.01 },
  { key: "dissolved_oxygen", label: "Dissolved oxygen", unit: "mg/L", min: 0, max: 25, step: 0.1 },
  { key: "suspended_solids", label: "Suspended solids", unit: "mg/L", min: 0, max: 200, step: 1 },
  { key: "ph", label: "pH", unit: "", min: 3, max: 11, step: 0.1 },
];

/**
 * A plausible warm July day on the Ayase, chosen to sit *inside* the training
 * range: the depth and velocity are the reach means the hydraulic model gives
 * at 12 m3/s, so the app does not open with an out-of-range warning already
 * showing.
 */
export const DEFAULT_STATE: EnvironmentalState = {
  water_temp: 24.5,
  air_temp: 26.0,
  discharge: 12.0,
  depth: 2.9,
  velocity: 0.41,
  dissolved_oxygen: 7.2,
  suspended_solids: 15,
  ph: 7.4,
  month: 7,
};

export const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * The training range covering a form field.
 *
 * A form field is not always a model feature: the model takes reach-averaged
 * `reach_depth`, while the caller supplies a point `depth` that the API maps
 * onto it. The aliases keep the warning attached to the control the user is
 * actually moving.
 */
const RANGE_ALIASES: Record<string, string[]> = {
  depth: ["depth", "reach_depth"],
  velocity: ["velocity", "reach_velocity"],
};

export function trainingRange(
  model: ModelInfo | undefined,
  key: string,
): [number, number] | null {
  const ranges = model?.training_ranges;
  if (!ranges) return null;
  for (const candidate of RANGE_ALIASES[key] ?? [key]) {
    const found = ranges[candidate];
    if (found && found.length === 2) return [found[0], found[1]];
  }
  return null;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

interface Props {
  state: EnvironmentalState;
  onChange: (state: EnvironmentalState) => void;
  model?: ModelInfo;
  disabled?: boolean;
}

export default function EnvironmentalInput({ state, onChange, model, disabled }: Props) {
  const used = new Set(model?.features ?? []);

  const set = (key: keyof EnvironmentalState, value: number) =>
    onChange({ ...state, [key]: value });

  return (
    <div>
      {FIELDS.map((field) => {
        const value = state[field.key] ?? field.min;
        const range = trainingRange(model, field.key);
        const outside = range !== null && (value < range[0] || value > range[1]);
        const inputId = `field-${field.key}`;
        const unused =
          model !== undefined &&
          !used.has(field.key) &&
          !(RANGE_ALIASES[field.key] ?? []).some((alias) => used.has(alias));

        return (
          <div className="field" key={field.key}>
            <div className="field-head">
              <label htmlFor={inputId}>
                {field.label}
                {unused && <span className="muted small"> · not used by this model</span>}
              </label>
              <span className={`field-value ${outside ? "out" : ""}`}>
                {value}
                {field.unit && ` ${field.unit}`}
                {outside && " · outside training"}
              </span>
            </div>
            <input
              id={inputId}
              type="range"
              min={field.min}
              max={field.max}
              step={field.step}
              value={value}
              disabled={disabled}
              aria-label={`${field.label}${field.unit ? ` (${field.unit})` : ""}`}
              onChange={(event) => set(field.key, Number(event.target.value))}
            />
            <div className="range-hint">
              <span>
                {range
                  ? `trained ${round(range[0])}–${round(range[1])}`
                  : "training range unknown"}
              </span>
              <span>
                {field.min}–{field.max} accepted
              </span>
            </div>
          </div>
        );
      })}

      <div className="field">
        <div className="field-head">
          <label htmlFor="field-month">Month</label>
          <span className="field-value">{MONTHS[(state.month ?? 1) - 1]}</span>
        </div>
        <select
          id="field-month"
          value={state.month ?? 7}
          disabled={disabled}
          onChange={(event) => set("month", Number(event.target.value))}
        >
          {MONTHS.map((name, index) => (
            <option key={name} value={index + 1}>
              {name}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
