/**
 * Analyze: the model's response surface, and what it is made of.
 */

import { useState } from "react";

import EnvironmentalInput, { DEFAULT_STATE } from "../components/EnvironmentalInput";
import InteractionHeatmap, {
  buildGrid,
  GRID_STEPS,
  linspace,
  type GridResult,
} from "../components/InteractionHeatmap";
import { ErrorBanner, ProvenanceBadge } from "../components/Provenance";
import TargetPicker from "../components/TargetPicker";
import { api, ApiError } from "../services/api";
import type { EnvironmentalState, ModelInfo, Target } from "../types";
import type { ModelState } from "../useModels";

export default function Analyze({ models }: { models: ModelState }) {
  const [base, setBase] = useState<EnvironmentalState>(DEFAULT_STATE);
  const [target, setTarget] = useState<Target>("dissolved_oxygen");
  const [grid, setGrid] = useState<GridResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const model = models.byTarget(target);

  const sweep = async () => {
    setBusy(true);
    setError(null);
    try {
      const temperatureRange = model?.training_ranges?.water_temp ?? [5, 32];
      const dischargeRange = model?.training_ranges?.discharge ?? [0.2, 74];
      const xValues = linspace(dischargeRange[0], dischargeRange[1], GRID_STEPS);
      const yValues = linspace(temperatureRange[0], temperatureRange[1], GRID_STEPS);

      const response = await api.batchPredict(
        buildGrid(base, xValues, yValues),
        target,
      );
      setGrid({
        xValues,
        yValues,
        predictions: response.predictions,
        unit: response.unit,
        outOfRangeCount: response.out_of_range_count,
      });
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <h1>Analyze</h1>
      <p className="muted prose">
        A {GRID_STEPS}×{GRID_STEPS} sweep across temperature and discharge, sent
        as a single batch request. It shows the shape of what the model learned —
        which is not the same as the shape of the river.
      </p>

      {error && <ErrorBanner message={error} />}

      <div className="split">
        <section className="card" aria-label="Sweep settings">
          <div className="card-title">
            <h2>Held constant</h2>
          </div>
          <TargetPicker
            value={target}
            onChange={setTarget}
            models={models.models}
            disabled={busy}
          />
          <p className="small muted" style={{ marginTop: "0.6rem" }}>
            Temperature and discharge are swept across the model's training range.
            Everything below is held fixed for every cell.
          </p>
          <EnvironmentalInput
            state={base}
            onChange={setBase}
            model={model}
            disabled={busy}
          />
          <button type="button" onClick={sweep} disabled={busy}>
            {busy ? "Sweeping…" : `Run ${GRID_STEPS * GRID_STEPS} predictions`}
          </button>
        </section>

        <div>
          <section className="card" aria-label="Response surface">
            <div className="card-title">
              <h2>Response surface</h2>
              {model && <ProvenanceBadge labels={model.labels} />}
            </div>
            {grid ? (
              <InteractionHeatmap grid={grid} target={target} />
            ) : (
              <p className="muted" style={{ margin: 0 }}>
                Run the sweep to fill the grid.
              </p>
            )}
          </section>

          {model && <ModelCard model={model} />}
        </div>
      </div>
    </>
  );
}

function ModelCard({ model }: { model: ModelInfo }) {
  const metrics = Object.entries(model.metrics);

  return (
    <section className="card" aria-label="Model detail">
      <div className="card-title">
        <h2>What is answering</h2>
        <ProvenanceBadge labels={model.labels} />
      </div>

      <div className="scroll-x">
        <table>
          <tbody>
            <tr>
              <th scope="row">Model</th>
              <td>{model.model_type}</td>
            </tr>
            <tr>
              <th scope="row">Training rows</th>
              <td className="num">{model.n_train.toLocaleString()}</td>
            </tr>
            <tr>
              <th scope="row">Features</th>
              <td className="small mono">{model.features.join(", ")}</td>
            </tr>
            {metrics.map(([name, value]) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td className={typeof value === "number" ? "num" : ""}>
                  {typeof value === "number" ? value.toFixed(3) : value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={`caveats ${model.labels === "SYNTHETIC" ? "synthetic" : ""}`}>
        <h3>Read with the number</h3>
        <ul>
          {model.caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
