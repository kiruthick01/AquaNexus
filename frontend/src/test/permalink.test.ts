/**
 * Scenario permalinks.
 *
 * Two properties matter. A link must survive the round trip, or sharing one
 * quietly changes the question. And a link must never carry an answer - the
 * reader has to get the current model's number with the current model's
 * caveats, not a figure frozen at the moment somebody pressed copy.
 */

import { describe, expect, it } from "vitest";

import { DEFAULT_STATE } from "../components/EnvironmentalInput";
import {
  decodeScenario,
  encodeScenario,
  scenarioUrl,
  type ScenarioLink,
} from "../services/permalink";

const FALLBACK: ScenarioLink = {
  target: "dissolved_oxygen",
  baseline: DEFAULT_STATE,
  modifications: { discharge: -0.6 },
  name: "Drought",
};

const decode = (query: string) =>
  decodeScenario(new URLSearchParams(query), FALLBACK);

describe("scenario permalinks", () => {
  it("round-trips a scenario", () => {
    const link: ScenarioLink = {
      target: "hsi",
      baseline: { ...DEFAULT_STATE, water_temp: 28.4, discharge: 3.2, month: 8 },
      modifications: { discharge: -0.45, water_temp: 0.1 },
      name: "Warm low flow",
    };

    expect(decodeScenario(encodeScenario(link), FALLBACK)).toEqual(link);
  });

  it("writes the whole baseline, not only what differs from the default", () => {
    // The default state is a chosen state and has moved before. A link that
    // leaned on it would mean something different after the next change.
    const params = encodeScenario(FALLBACK);
    expect(params.get("water_temp")).toBe(String(DEFAULT_STATE.water_temp));
    expect(params.get("ph")).toBe(String(DEFAULT_STATE.ph));
    expect(params.get("month")).toBe(String(DEFAULT_STATE.month));
  });

  it("carries no prediction, so opening it asks the model again", () => {
    const query = encodeScenario(FALLBACK).toString();
    for (const forbidden of ["prediction", "change", "baseline_prediction", "unit"]) {
      expect(query).not.toContain(forbidden);
    }
  });

  it("omits a change of zero", () => {
    const query = encodeScenario({
      ...FALLBACK,
      modifications: { discharge: -0.6, water_temp: 0 },
    }).toString();
    expect(query).toContain("m.discharge=-0.6");
    expect(query).not.toContain("m.water_temp");
  });

  it("falls back field by field when the link is partial", () => {
    const link = decode("target=hsi&water_temp=30");
    expect(link.target).toBe("hsi");
    expect(link.baseline.water_temp).toBe(30);
    expect(link.baseline.discharge).toBe(DEFAULT_STATE.discharge);
    expect(link.modifications).toEqual(FALLBACK.modifications);
    expect(link.name).toBe(FALLBACK.name);
  });

  it("ignores what it cannot read rather than failing", () => {
    const link = decode("target=rainfall&water_temp=hot&m.gravity=-0.5&ph=");
    expect(link.target).toBe("dissolved_oxygen");
    expect(link.baseline.water_temp).toBe(DEFAULT_STATE.water_temp);
    expect(link.baseline.ph).toBe(DEFAULT_STATE.ph);
    expect(link.modifications).toEqual(FALLBACK.modifications);
  });

  it("clamps to the physical limits the API enforces", () => {
    const link = decode("water_temp=900&discharge=-5&m.discharge=-4");
    expect(link.baseline.water_temp).toBe(45);
    expect(link.baseline.discharge).toBe(0);
    expect(link.modifications.discharge).toBe(-1);
  });

  it("keeps a state outside the training range, because the API answers those", () => {
    // Out of *training* range is a legitimate question - the API flags it and
    // answers it. Only the physical limits are clamped.
    const link = decode("discharge=110");
    expect(link.baseline.discharge).toBe(110);
  });

  it("truncates a name rather than letting it become the page", () => {
    const link = decode(`name=${"x".repeat(200)}`);
    expect(link.name).toHaveLength(60);
  });

  it("builds an absolute URL for the clipboard", () => {
    const url = scenarioUrl(FALLBACK, "https://aquanexus.example", "/scenarios");
    expect(url.startsWith("https://aquanexus.example/scenarios?")).toBe(true);
    expect(url).toContain("target=dissolved_oxygen");
  });
});
