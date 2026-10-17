import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

import App from "../App";
import { health, mockFetch, models, prediction, station } from "./fixtures";

const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );

afterEach(() => vi.unstubAllGlobals());

describe("overview", () => {
  it("leads with held-out accuracy from the API, not hard-coded numbers", async () => {
    mockFetch({ "/health": health, "/models": models });
    renderAt("/");
    expect(await screen.findByText("R² 0.72")).toBeInTheDocument();
    expect(screen.getByText("R² 0.48")).toBeInTheDocument();
    expect(screen.getByText(/never seen/)).toBeInTheDocument();
  });

  it("explains a degraded backend instead of showing empty pages", async () => {
    mockFetch({
      "/health": { ...health, status: "degraded", models_loaded: [], detail: "no model bundles" },
    });
    renderAt("/");
    expect(await screen.findByRole("alert")).toHaveTextContent("no model bundles");
  });
});

describe("models page", () => {
  it("shows each model with its caveat and interval widths", async () => {
    mockFetch({ "/health": health, "/models": models });
    renderAt("/models");
    expect(await screen.findByRole("region", { name: "geo" })).toBeInTheDocument();
    expect(screen.getAllByText(/scored on unseen stations/)).toHaveLength(2);
    expect(screen.getAllByText(/2\.3 \/ 1\.7 \/ 1\.4/)).toHaveLength(2);
  });
});

describe("predict page", () => {
  it("sends the form and shows the interval and the missing inputs", async () => {
    const calls = mockFetch({ "/health": health, "/models": models, "/predict": prediction });
    renderAt("/predict");
    await userEvent.click(await screen.findByRole("button", { name: "Predict" }));

    expect(await screen.findByText("6.11")).toBeInTheDocument();
    expect(screen.getByText(/3\.91 to 8\.32 mg\/L/)).toBeInTheDocument();
    expect(screen.getByText(/rain_1d/)).toBeInTheDocument();

    const sent = calls.find((c) => c.url.endsWith("/predict"))!;
    const body = JSON.parse(String(sent.init?.body));
    expect(body.location).toEqual({ latitude: 35.8, longitude: 139.8 });
    expect(body.weather.air_temp).toBe(25);
    expect(body.explain).toBe(true);
  });

  it("uses a searched station instead of coordinates", async () => {
    const calls = mockFetch({
      "/health": health,
      "/models": models,
      "/stations": [station],
      "/predict": prediction,
    });
    renderAt("/predict");
    await userEvent.type(await screen.findByLabelText(/Station/), "綾瀬");
    await userEvent.click(screen.getByRole("button", { name: "Search" }));
    await userEvent.click(await screen.findByRole("button", { name: /畷橋/ }));
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => expect(calls.some((c) => c.url.endsWith("/predict"))).toBe(true));
    const sent = calls.find((c) => c.url.endsWith("/predict"))!;
    const body = JSON.parse(String(sent.init?.body));
    expect(body.station_id).toBe(1110100);
    expect(body.location).toBeUndefined();
  });

  it("shows the API's own error text", async () => {
    mockFetch({
      "/health": health,
      "/models": models,
      "/predict": new Error("model 'full' is not loaded"),
    });
    renderAt("/predict");
    await userEvent.click(await screen.findByRole("button", { name: "Predict" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("not loaded");
  });
});
