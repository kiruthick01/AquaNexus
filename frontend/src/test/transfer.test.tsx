/**
 * The transfer page.
 *
 * The thing being tested is editorial as much as functional: this result is
 * easy to render dishonestly in either direction. Showing the pooled R² -0.081
 * as the headline hides that the model transfers where it has evidence; showing
 * only the in-range +0.336 hides that a third of the river is outside it. Both
 * have to be on the page, and the split has to lead.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import TransferScale, { marksFor } from "../components/TransferScale";
import * as fixtures from "./fixtures";

function stubApi(routes: Record<string, unknown>, status = 200) {
  const spy = vi.fn(async (url: string) => {
    const path = new URL(url).pathname;
    const body = routes[path];
    if (body === undefined) {
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: `no stub for ${path}` }),
      };
    }
    return { ok: status < 400, status, json: async () => body };
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

const renderApp = (path = "/transfer") =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );

const withHoldout = () => ({
  "/health": fixtures.health,
  "/models": [fixtures.oxygenModel, fixtures.hsiModel],
  "/holdout": fixtures.holdout,
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Transfer", () => {
  it("leads with the split rather than the pooled score", async () => {
    stubApi(withHoldout());
    renderApp();

    const headline = await screen.findByText(/Inside the ranges it was fitted on/);
    expect(headline).toBeInTheDocument();
    expect(headline.textContent).toContain("+0.336");
    expect(headline.textContent).toContain("+0.394");
  });

  it("still states the pooled score, which is the unflattering one", async () => {
    stubApi(withHoldout());
    renderApp();

    // -0.081 must appear: omitting it would be the flattering half-truth.
    await waitFor(() =>
      expect(screen.getAllByText(/-0\.081/).length).toBeGreaterThan(0),
    );
  });

  it("names the river as never seen and the model as unchanged", async () => {
    stubApi(withHoldout());
    renderApp();

    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading).toHaveTextContent(/never seen/i);
    expect(screen.getByText("unchanged")).toBeInTheDocument();
    expect(screen.getAllByText(/中川/).length).toBeGreaterThan(0);
  });

  it("shows the references that separate a bad transfer from a hard river", async () => {
    stubApi(withHoldout());
    renderApp();

    const table = await screen.findByLabelText("Scored against references");
    expect(within(table).getByText("mean of this river")).toBeInTheDocument();
    expect(within(table).getByText("persistence")).toBeInTheDocument();
    expect(within(table).getByText("trained on this river")).toBeInTheDocument();
  });

  it("marks the extrapolating station and not the in-range one", async () => {
    stubApi(withHoldout());
    renderApp();

    const table = await screen.findByLabelText("Per station");
    const outside = within(table).getByText("46八条橋").closest("tr")!;
    const inside = within(table).getByText("51道橋").closest("tr")!;

    expect(within(outside).getByText("extrapolating")).toBeInTheDocument();
    expect(within(inside).getByText("in range")).toBeInTheDocument();
  });

  it("points at the document when the API has no holdout result", async () => {
    stubApi({ "/health": fixtures.health, "/models": [fixtures.oxygenModel] });
    renderApp();

    // The stub answers 404 for /holdout; a real degraded API answers 503. Both
    // must leave the reader knowing where the written result lives.
    expect(await screen.findByText(/docs\/HOLDOUT_RIVER\.md/)).toBeInTheDocument();
    expect(screen.getByText(/scripts\/holdout_river\.py/)).toBeInTheDocument();
  });

  it("is reachable from the masthead", async () => {
    stubApi(withHoldout());
    renderApp("/");

    const link = await screen.findByRole("link", { name: "Transfer" });
    expect(link).toHaveAttribute("href", "/transfer");
  });
});

describe("TransferScale", () => {
  it("places the two regimes either side of the datum", () => {
    const { marks, floor } = marksFor(fixtures.holdout.by_evidence, 0.394);
    const inside = marks.find((m) => m.label.startsWith("inside"))!;
    const outside = marks.find((m) => m.label === "extrapolating")!;

    expect(inside.r2).toBeGreaterThan(0);
    expect(outside.r2).toBeLessThan(0);
    // The floor has to reach past the worst mark, or it is drawn off the staff.
    expect(floor).toBeLessThanOrEqual(outside.r2);
  });

  it("draws the home score as a reference rather than a result", () => {
    const { marks } = marksFor(fixtures.holdout.by_evidence, 0.394);
    const home = marks.find((m) => m.reference)!;

    expect(home.r2).toBeCloseTo(0.394, 3);
    // Drawn as a datum, not as one of the transfer's own two results.
    expect(marks.filter((m) => !m.reference)).toHaveLength(2);
  });

  it("describes itself for a reader who cannot see the figure", () => {
    const { marks, floor } = marksFor(fixtures.holdout.by_evidence, 0.394);
    render(<TransferScale marks={marks} floor={floor} />);

    const figure = screen.getByRole("img");
    expect(figure.getAttribute("aria-label")).toContain("0.336");
    expect(figure.getAttribute("aria-label")).toContain("-0.903");
  });

  it("omits the home reference when no model is loaded to compare against", () => {
    const { marks } = marksFor(fixtures.holdout.by_evidence, undefined);
    expect(marks.some((m) => m.reference)).toBe(false);
    expect(marks).toHaveLength(2);
  });
});
