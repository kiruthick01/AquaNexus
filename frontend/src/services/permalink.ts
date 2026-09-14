/**
 * Scenario permalinks: the inputs of a what-if, carried in the URL.
 *
 * A scenario is worth sending to somebody, and until now the only way to do
 * that was a screenshot of a number - which is exactly the thing this project
 * argues against, because a screenshot separates the number from its caveats
 * and from the model that produced it.
 *
 * What the link carries is the **inputs**: the target, the baseline state, the
 * fractional changes and the name. Not the answer. Opening it re-runs the
 * scenario against whatever the API is serving now, so the reader gets the
 * current model's answer with the current model's provenance attached, and a
 * link shared before a retrain cannot quietly show yesterday's number as if it
 * were today's.
 *
 * The whole baseline is written out even where it equals the default. The
 * default is a chosen state, not a constant of nature - it moved once already
 * when the geometry was corrected - and a link that omitted those fields would
 * silently mean something different after the next such change.
 */

import { FIELDS } from "../components/EnvironmentalInput";
import type { EnvironmentalState, Target } from "../types";

/** Prefix for a fractional change, so `discharge` and its delta never collide. */
const MODIFICATION_PREFIX = "m.";

/** Slider bounds in ScenarioBuilder; a link cannot ask for more than the UI can. */
const MODIFICATION_MIN = -1;
const MODIFICATION_MAX = 2;

/** Long enough for a sentence fragment, short enough not to break the heading. */
const MAX_NAME_LENGTH = 60;

const TARGETS: Target[] = ["dissolved_oxygen", "hsi"];

export interface ScenarioLink {
  target: Target;
  baseline: EnvironmentalState;
  modifications: Record<string, number>;
  name: string;
}

const BOUNDS = new Map(FIELDS.map((field) => [field.key as string, field]));

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Read one numeric parameter, or undefined if it is absent or unusable.
 *
 * Clamping is to the *physical* limits the API enforces, not to the training
 * range. A state outside the training range is a legitimate question - the API
 * answers it and flags it, and the form allows it - so a link may carry one.
 * A state outside the physical limits is a 422, and clamping is friendlier than
 * handing the reader a broken page.
 */
function readNumber(raw: string | null, min: number, max: number): number | undefined {
  if (raw === null || raw.trim() === "") return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) return undefined;
  return clamp(value, min, max);
}

export function encodeScenario(link: ScenarioLink): URLSearchParams {
  const params = new URLSearchParams();
  params.set("target", link.target);

  for (const field of FIELDS) {
    const value = link.baseline[field.key];
    if (typeof value === "number" && Number.isFinite(value)) {
      params.set(field.key, String(value));
    }
  }
  if (typeof link.baseline.month === "number") {
    params.set("month", String(link.baseline.month));
  }

  for (const [key, fraction] of Object.entries(link.modifications)) {
    // A zero change is the absence of a change: writing it would put the reader
    // in front of a slider that says 0% and a URL that implies it was chosen.
    if (Number.isFinite(fraction) && fraction !== 0) {
      params.set(MODIFICATION_PREFIX + key, String(fraction));
    }
  }

  const name = link.name.trim();
  if (name) params.set("name", name.slice(0, MAX_NAME_LENGTH));
  return params;
}

/**
 * Rebuild a scenario from a URL, falling back field by field.
 *
 * Every part is optional and every unreadable part is ignored rather than
 * fatal: a link that has been truncated by a chat client, or hand-edited, opens
 * on the defaults instead of an error page.
 */
export function decodeScenario(
  params: URLSearchParams,
  fallback: ScenarioLink,
): ScenarioLink {
  const rawTarget = params.get("target");
  const target = TARGETS.find((candidate) => candidate === rawTarget) ?? fallback.target;

  const baseline: EnvironmentalState = { ...fallback.baseline };
  for (const field of FIELDS) {
    const value = readNumber(params.get(field.key), field.min, field.max);
    if (value !== undefined) baseline[field.key] = value;
  }
  const month = readNumber(params.get("month"), 1, 12);
  if (month !== undefined) baseline.month = Math.round(month);

  const modifications: Record<string, number> = {};
  for (const [key, raw] of params.entries()) {
    if (!key.startsWith(MODIFICATION_PREFIX)) continue;
    const field = key.slice(MODIFICATION_PREFIX.length);
    if (!BOUNDS.has(field)) continue;
    const fraction = readNumber(raw, MODIFICATION_MIN, MODIFICATION_MAX);
    if (fraction !== undefined) modifications[field] = fraction;
  }

  const name = (params.get("name") ?? "").trim().slice(0, MAX_NAME_LENGTH);

  return {
    target,
    baseline,
    modifications: Object.keys(modifications).length
      ? modifications
      : fallback.modifications,
    name: name || fallback.name,
  };
}

/** The absolute URL for a scenario, for copying to a clipboard. */
export function scenarioUrl(link: ScenarioLink, origin: string, path: string): string {
  return `${origin}${path}?${encodeScenario(link).toString()}`;
}
