/**
 * Overview: what the models are, and how well they did on stations they never saw.
 */

import { Link } from "react-router-dom";

import type { ModelState } from "../useModels";

const fmt = (value: number | undefined, digits = 2) =>
  value === undefined ? "–" : value.toFixed(digits);

export default function Home({ models }: { models: ModelState }) {
  const full = models.byMode("full");
  const geo = models.byMode("geo");

  return (
    <>
      <h1>Dissolved oxygen in Japanese rivers</h1>
      <p className="muted prose">
        Gradient-boosted models trained on about 565,000 real monitoring samples from 5,883 river
        stations across all 47 prefectures. Every accuracy figure below comes from stations the model
        had never seen, in years after the ones it trained on.
      </p>

      <section className="card" aria-label="Held-out accuracy">
        <div className="card-title">
          <h2>How well does it do?</h2>
        </div>
        {geo && full ? (
          <div className="facts">
            <div>
              <div className="stat-label">New station, weather only</div>
              <div className="stat-value">R² {fmt(geo.held_out_strict.r2)}</div>
              <div className="small muted">error {fmt(geo.held_out_strict.mae)} mg/L</div>
            </div>
            <div>
              <div className="stat-label">Station with history</div>
              <div className="stat-value">R² {fmt(full.held_out_strict.r2)}</div>
              <div className="small muted">error {fmt(full.held_out_strict.mae)} mg/L</div>
            </div>
            <div>
              <div className="stat-label">Low-oxygen events caught</div>
              <div className="stat-value">{fmt((full.held_out_strict.recall_low ?? 0) * 100, 0)}%</div>
              <div className="small muted">
                of samples below 5 mg/L; {fmt((full.held_out_strict.precision_low ?? 0) * 100, 0)}% of
                alarms are right
              </div>
            </div>
          </div>
        ) : (
          <p className="muted">Model metadata is not available.</p>
        )}
        <p className="small muted" style={{ marginBottom: 0 }}>
          Low oxygen is the hard part: only 2% of samples are below 5 mg/L, and monthly sampling
          misses most events. See <Link to="/models">Models</Link> for every number and its caveats.
        </p>
      </section>

      <p>
        <Link to="/predict">Make a prediction →</Link>
      </p>
    </>
  );
}
