/**
 * API client behaviour, with `fetch` stubbed.
 *
 * The error paths matter more than the happy ones here: the backend answers 503
 * with a reason when a model or its explainer is missing, and a client that
 * swallows that reason turns a fixable setup problem into a blank screen.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { api, ApiError, API_BASE } from "../services/api";
import * as fixtures from "./fixtures";

function stubFetch(response: Partial<Response> & { json?: () => Promise<unknown> }) {
  const spy = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({}),
    ...response,
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("posts a prediction request to the configured base URL", async () => {
    const spy = stubFetch({ json: async () => fixtures.prediction });

    const result = await api.predict({ water_temp: 24.5 }, "dissolved_oxygen");

    expect(result.prediction).toBe(4.95);
    const [url, init] = spy.mock.calls[0];
    expect(url).toBe(`${API_BASE}/predict`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      state: { water_temp: 24.5 },
      target: "dissolved_oxygen",
      explain: false,
    });
  });

  it("surfaces the API's own detail on a 503", async () => {
    stubFetch({
      ok: false,
      status: 503,
      json: async () => ({ detail: "no background sample for dissolved_oxygen" }),
    });

    await expect(api.explain({}, "dissolved_oxygen")).rejects.toThrow(
      /no background sample/,
    );
  });

  it("flattens FastAPI validation errors into one message", async () => {
    stubFetch({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [{ loc: ["body", "state", "water_temp"], msg: "less than -2" }],
      }),
    });

    await expect(api.predict({}, "dissolved_oxygen")).rejects.toThrow(
      /state\.water_temp: less than -2/,
    );
  });

  it("says which service is unreachable when fetch itself fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const error = await api.health().catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toContain(API_BASE);
  });

  it("sends a batch as one request", async () => {
    const spy = stubFetch({
      json: async () => ({ ...fixtures.prediction, n: 2, predictions: [1, 2] }),
    });

    await api.batchPredict([{ water_temp: 10 }, { water_temp: 20 }], "hsi");

    expect(spy).toHaveBeenCalledTimes(1);
    const body = JSON.parse(spy.mock.calls[0][1].body);
    expect(body.states).toHaveLength(2);
    expect(body.target).toBe("hsi");
  });
});
