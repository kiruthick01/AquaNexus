/**
 * The provenance contract.
 *
 * These tests exist to stop a refactor quietly dropping the one thing this
 * project cannot afford to lose: a number rendered without the fact that its
 * labels were generated. They assert on what a reader sees, not on props.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ExplainabilityPanel from "../components/ExplainabilityPanel";
import PredictionCard from "../components/PredictionCard";
import * as fixtures from "./fixtures";

describe("PredictionCard", () => {
  it("shows the value, unit and interpretation", () => {
    render(<PredictionCard result={fixtures.prediction} />);
    expect(screen.getByText("4.95")).toBeInTheDocument();
    expect(screen.getByText("mg/L")).toBeInTheDocument();
    expect(screen.getByText(/hypoxic stress likely/)).toBeInTheDocument();
  });

  it("marks a synthetic-label model as synthetic", () => {
    render(<PredictionCard result={fixtures.syntheticPrediction} />);
    const card = screen.getByRole("region", { name: "Prediction" });
    // The badge, and separately the caveat text, both say so.
    expect(within(card).getByText("⚠ Synthetic labels")).toBeInTheDocument();
    expect(
      within(card).getByText(/SYNTHETIC LABELS\. HSI is generated/),
    ).toBeInTheDocument();
  });

  it("does not label the observed model as synthetic", () => {
    render(<PredictionCard result={fixtures.prediction} />);
    expect(screen.queryByText(/synthetic/i)).not.toBeInTheDocument();
    expect(screen.getByText("Observed labels")).toBeInTheDocument();
  });

  it("always renders the caveats that came with the prediction", () => {
    render(<PredictionCard result={fixtures.prediction} />);
    for (const caveat of fixtures.prediction.caveats) {
      expect(screen.getByText(caveat)).toBeInTheDocument();
    }
  });

  it("reports a missing uncertainty rather than inventing one", () => {
    render(<PredictionCard result={fixtures.prediction} />);
    expect(screen.getByText("not available")).toBeInTheDocument();
  });

  it("flags an input outside the training range", () => {
    render(<PredictionCard result={fixtures.outOfRangePrediction} />);
    expect(screen.getByText(/Outside the training range/i)).toBeInTheDocument();
    expect(screen.getByText(/discharge/)).toBeInTheDocument();
    expect(screen.getByText(/900/)).toBeInTheDocument();
  });
});

describe("ExplainabilityPanel", () => {
  it("warns about collinear features before showing the chart", () => {
    const { container } = render(
      <ExplainabilityPanel explanation={fixtures.explanation} />,
    );
    const warning = screen.getByRole("note");
    const chart = screen.getByRole("img", {
      name: /per-feature shap contributions/i,
    });

    expect(warning).toHaveTextContent(/2 feature pairs move together/i);
    // Document order matters: the caveat has to be read before the ranking.
    expect(
      warning.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(container.querySelector("svg")).toBeTruthy();
  });

  it("renders one bar per contributing feature", () => {
    render(<ExplainabilityPanel explanation={fixtures.explanation} />);
    const chart = screen.getByRole("img", {
      name: /per-feature shap contributions/i,
    });
    for (const row of fixtures.explanation.contributions) {
      // Feature names also appear in the collinearity note, so scope the query.
      expect(within(chart).getByText(new RegExp(row.feature))).toBeInTheDocument();
    }
  });

  it("states that contributions sum to the gap from the baseline", () => {
    render(<ExplainabilityPanel explanation={fixtures.explanation} />);
    const gap = fixtures.explanation.prediction - fixtures.explanation.baseline;
    expect(
      screen.getByText(new RegExp(`${gap.toFixed(3)}`)),
    ).toBeInTheDocument();
  });
});
