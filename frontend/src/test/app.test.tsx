/**
 * The app end to end, against a stubbed backend.
 *
 * Covers the two flows a reviewer will actually try - predict, and predict with
 * an explanation - plus the degraded case, which is what a first-time visitor
 * hits when the API is running but no models were trained.
 */

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import * as fixtures from "./fixtures";

/** Routes stubbed fetches by path, so a test only declares what it needs. */
function stubApi(routes: Record<string, unknown>, status = 200) {
  const spy = vi.fn(async (url: string, init?: RequestInit) => {
    void init;
    const path = new URL(url).pathname;
    const body = routes[path];
    if (body === undefined) {
      return { ok: false, status: 404, json: async () => ({ detail: `no stub for ${path}` }) };
    }
    return { ok: status < 400, status, json: async () => body };
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

const renderApp = (path = "/") =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("summarises both models on the overview, each with its provenance", async () => {
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
    });

    renderApp("/");

    expect(await screen.findByText("Dissolved oxygen")).toBeInTheDocument();
    expect(screen.getByText("Habitat index")).toBeInTheDocument();
    expect(screen.getByText("⚠ Synthetic labels")).toBeInTheDocument();
    expect(screen.getByText("Observed labels")).toBeInTheDocument();
  });

  it("explains itself when the API has no models loaded", async () => {
    const degraded = {
      ...fixtures.health,
      status: "degraded" as const,
      models_loaded: [],
      detail: "no manifest at /app/data/models/manifest.json; run scripts/train_models.py",
    };
    stubApi({ "/health": degraded });

    renderApp("/");

    const banner = await screen.findByRole("alert");
    expect(banner).toHaveTextContent(/not answering with models/i);
    expect(banner).toHaveTextContent(/run scripts\/train_models\.py/);
  });

  it("predicts from the form and shows the result with its caveats", async () => {
    const user = userEvent.setup();
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
      "/predict": fixtures.prediction,
    });

    renderApp("/predict");
    await screen.findByLabelText(/Water temperature/);

    await user.click(screen.getByRole("button", { name: "Predict" }));

    const card = await screen.findByRole("region", { name: "Prediction" });
    expect(within(card).getByText("4.95")).toBeInTheDocument();
    expect(
      within(card).getByText(/Barely beats a persistence baseline - 0.009 R2/),
    ).toBeInTheDocument();
  });

  it("fetches an explanation only when asked for one", async () => {
    const user = userEvent.setup();
    const spy = stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
      "/predict": fixtures.prediction,
      "/explain": fixtures.explanation,
    });

    renderApp("/predict");
    await screen.findByLabelText(/Water temperature/);

    await user.click(screen.getByRole("button", { name: "Predict" }));
    await screen.findByRole("region", { name: "Prediction" });
    expect(spy.mock.calls.map(([url]) => new URL(url).pathname)).not.toContain(
      "/explain",
    );

    await user.click(screen.getByRole("button", { name: /Predict and explain/ }));
    expect(await screen.findByRole("region", { name: "Explanation" })).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/move together/i);
  });

  it("shows the API's message when a request fails", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const path = new URL(url).pathname;
        if (path === "/health") return { ok: true, status: 200, json: async () => fixtures.health };
        if (path === "/models")
          return { ok: true, status: 200, json: async () => [fixtures.oxygenModel] };
        return {
          ok: false,
          status: 503,
          json: async () => ({ detail: "explanation unavailable: no background" }),
        };
      }),
    );

    renderApp("/predict");
    await screen.findByLabelText(/Water temperature/);
    await user.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/explanation unavailable/),
    );
  });

  it("keeps the synthetic warning when the habitat model is selected", async () => {
    const user = userEvent.setup();
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
      "/predict": fixtures.syntheticPrediction,
    });

    renderApp("/predict");
    await screen.findByLabelText(/Water temperature/);

    await user.click(screen.getByRole("button", { name: "Habitat index" }));
    expect(
      screen.getByText(/measures recovery of a function this project wrote/),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Predict" }));
    const card = await screen.findByRole("region", { name: "Prediction" });
    expect(within(card).getByText("⚠ Synthetic labels")).toBeInTheDocument();
  });

  it("runs a scenario and states that it is not a simulation", async () => {
    const user = userEvent.setup();
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
      "/scenario_run": fixtures.scenario,
    });

    renderApp("/scenarios");
    await screen.findByRole("region", { name: "Scenario builder" });

    await user.click(screen.getByRole("button", { name: "Run scenario" }));

    const result = await screen.findByRole("region", { name: "Scenario result" });
    expect(within(result).getByText(/6.92/)).toBeInTheDocument();
    expect(
      within(result).getByText(/model sensitivity, not a simulation/i),
    ).toBeInTheDocument();
  });

  it("opens a shared scenario with the link's state, not the defaults", async () => {
    const user = userEvent.setup();
    const spy = stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
      "/scenario_run": fixtures.scenario,
    });

    renderApp(
      "/scenarios?target=hsi&water_temp=31.5&discharge=2.5&month=8" +
        "&m.discharge=-0.3&name=Late%20summer",
    );
    await screen.findByRole("region", { name: "Scenario builder" });

    expect(screen.getByLabelText("Scenario name")).toHaveValue("Late summer");
    expect(screen.getByLabelText("Discharge change")).toHaveValue("-0.3");

    await user.click(screen.getByRole("button", { name: "Run scenario" }));

    // The request must carry what the link said, not what the form defaults to:
    // a link that renders correctly and asks a different question is worse than
    // one that fails.
    await waitFor(() => {
      const posted = spy.mock.calls.find((call) => call[0].endsWith("/scenario_run"));
      expect(posted).toBeDefined();
      const body = JSON.parse(posted![1]!.body as string);
      expect(body.target).toBe("hsi");
      expect(body.scenario_name).toBe("Late summer");
      expect(body.baseline.water_temp).toBe(31.5);
      expect(body.baseline.discharge).toBe(2.5);
      expect(body.modifications).toEqual({ discharge: -0.3 });
    });
  });

  it("copies a link to the scenario rather than to its answer", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn(async (_text: string) => {});
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
    });

    renderApp("/scenarios");
    await screen.findByRole("region", { name: "Scenario builder" });

    await user.click(screen.getByRole("button", { name: "Copy link" }));

    await waitFor(() => expect(writeText).toHaveBeenCalled());
    const url = writeText.mock.calls[0]![0];
    expect(url).toContain("/scenarios?");
    expect(url).toContain("m.discharge=-0.6");
    expect(url).not.toContain("prediction");
  });

  it("shows the link when the clipboard never answers", async () => {
    // Regression, found in a real browser: Chrome defers clipboard writes while
    // the document is hidden, and writeText then never settles - so awaiting it
    // left the button doing nothing at all, with no error to catch. Real timers
    // here: the component races a real 1.2 s timeout, and faking them fights
    // userEvent for the same clock.
    const user = userEvent.setup();
    vi.stubGlobal("navigator", {
      ...navigator,
      clipboard: { writeText: () => new Promise(() => {}) },
    });
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
    });

    renderApp("/scenarios");
    await screen.findByRole("region", { name: "Scenario builder" });

    await user.click(screen.getByRole("button", { name: "Copy link" }));

    const field = await screen.findByLabelText("Scenario link", {}, { timeout: 4000 });
    expect((field as HTMLInputElement).value).toContain("m.discharge=-0.6");
  }, 10000);

  it("warns before running a drought the model cannot answer", async () => {
    stubApi({
      "/health": fixtures.health,
      "/models": [fixtures.oxygenModel, fixtures.hsiModel],
    });

    renderApp("/scenarios");
    // The default baseline is 12 m3/s and the default preset is -60%, which
    // lands at 4.8 - above the threshold, so no warning yet.
    await screen.findByRole("region", { name: "Scenario builder" });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    // A range input takes a value change rather than typing; -90% of 12 m3/s
    // is 1.2, which is inside the band the model gets wrong.
    fireEvent.change(screen.getByLabelText("Discharge change"), {
      target: { value: "-0.9" },
    });

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(/not trustworthy/i),
    );
  });
});
