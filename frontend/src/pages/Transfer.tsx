/**
 * Transfer: the model on a river it was never trained on.
 *
 * Every other generalisation number in this dashboard is *within* the Ayase —
 * four stations held out one at a time, sharing a channel, a catchment and a
 * sampling programme. This page is the one place a reader can see what happened
 * when the finished model met the Naka, a catchment reserved before any of it
 * was built.
 *
 * The pooled score is R² -0.081, and showing it as the headline would be the
 * dishonest version of this result in one direction — the model does transfer,
 * inside the range it was fitted on. Showing only the in-range score would be
 * dishonest in the other. So the split leads, the pooled figure is stated
 * plainly underneath, and the per-station table shows which stations are which.
 */

import { useEffect, useState } from "react";

import { CaveatList, ErrorBanner } from "../components/Provenance";
import TransferScale, { marksFor } from "../components/TransferScale";
import { api, ApiError } from "../services/api";
import type { HoldoutResponse, TransferScore, TransferStation } from "../types";

export default function Transfer() {
  const [evidence, setEvidence] = useState<HoldoutResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await api.holdout();
        if (!cancelled) setEvidence(result);
      } catch (caught) {
        if (!cancelled) {
          setError(
            caught instanceof ApiError ? caught.message : "could not reach the API",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <p className="spinner">Loading the held-out river…</p>;
  if (error || !evidence) {
    return (
      <>
        <h1>Transfer</h1>
        <ErrorBanner message={error ?? "no holdout result available"} />
        <p className="muted prose">
          The result is written up in <code>docs/HOLDOUT_RIVER.md</code>{" "}
          regardless. To serve it here, run{" "}
          <code>python scripts/holdout_river.py</code>.
        </p>
      </>
    );
  }

  const homeR2 = evidence.home_metrics.r2;
  const { marks, floor } = marksFor(
    evidence.by_evidence,
    typeof homeR2 === "number" ? homeR2 : undefined,
  );
  const pooled = evidence.pooled[0];
  const ceiling = evidence.pooled.find((row) => row.label === "trained on this river");

  return (
    <>
      <h1>
        A river the model has never seen
      </h1>
      <p className="muted prose">
        The {evidence.river} ({evidence.river_ja}) was reserved before any of this
        was built and never touched until the model was finished. The{" "}
        {evidence.trained_on} model was then applied <strong>unchanged</strong> —
        no refitting, no recalibration — to {evidence.n} observations across{" "}
        {evidence.n_stations} stations.
      </p>

      <section className="card" aria-label="Transfer result">
        <div className="card-title">
          <h2>Where it lands</h2>
          <span className="small muted">
            {evidence.n} observations · {evidence.river_ja}
          </span>
        </div>

        <p className="interpretation prose">{evidence.headline}</p>

        <TransferScale marks={marks} floor={floor} />

        <p className="small muted prose" style={{ marginTop: "0.6rem" }}>
          Pooled over the whole river the transfer scores R²{" "}
          <span className="mono">{signed(pooled.r2)}</span> — below the datum. That
          number averages the two marks above, which is why it is not the headline:
          it describes neither regime.{" "}
          {ceiling && (
            <>
              A model of the same specification fitted to this river reaches{" "}
              <span className="mono">{signed(ceiling.r2)}</span>, so the gap is the
              cost of transfer rather than a difficult river.
            </>
          )}
        </p>
      </section>

      <div className="split">
        <section className="card" aria-label="Scored against references">
          <div className="card-title">
            <h2>Against what</h2>
          </div>
          <p className="small muted">
            Three references, all computed on the held-out river itself. Without
            them, “the model transfers badly” and “this river is hard” cannot be
            told apart.
          </p>
          <ScoreTable rows={evidence.pooled} />
        </section>

        <section className="card" aria-label="Split by evidence">
          <div className="card-title">
            <h2>Split by evidence</h2>
          </div>
          <p className="small muted">
            A row counts as in range only if every feature it supplies falls
            inside the range that feature took in training. These are the same
            rows every prediction flags as out of range — this is what that flag
            is worth.
          </p>
          <ScoreTable rows={evidence.by_evidence} />
        </section>
      </div>

      <section className="card" aria-label="Per station">
        <div className="card-title">
          <h2>Station by station</h2>
        </div>
        <p className="small muted prose">
          The stations are not interchangeable. One sub-reach carries flows the
          model was fitted on; the other carries more than it has ever seen, and
          it holds a quarter of the observations.
        </p>
        <StationTable stations={evidence.by_station} />
      </section>

      <CaveatList caveats={evidence.caveats} />

      <p className="small muted" style={{ marginTop: "1rem" }}>
        Generated by <code>scripts/holdout_river.py</code>; written up in{" "}
        <code>{evidence.document}</code>.
      </p>
    </>
  );
}

function ScoreTable({ rows }: { rows: TransferScore[] }) {
  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th scope="col"> </th>
            <th scope="col" className="num">
              n
            </th>
            <th scope="col" className="num">
              RMSE
            </th>
            <th scope="col" className="num">
              R²
            </th>
            <th scope="col" className="num">
              bias
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td className="num">{row.n}</td>
              <td className="num">{row.rmse.toFixed(3)}</td>
              <td className={`num ${row.r2 < 0 ? "below" : ""}`}>{signed(row.r2)}</td>
              <td className="num">{signed(row.bias)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StationTable({ stations }: { stations: TransferStation[] }) {
  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th scope="col">station</th>
            <th scope="col">sub-reach</th>
            <th scope="col" className="num">
              n
            </th>
            <th scope="col" className="num">
              mean Q
            </th>
            <th scope="col" className="num">
              R²
            </th>
            <th scope="col" className="num">
              bias
            </th>
            <th scope="col">evidence</th>
          </tr>
        </thead>
        <tbody>
          {stations.map((station) => (
            <tr key={station.station}>
              <th scope="row">{station.station}</th>
              <td className="small muted">{station.sub_reach}</td>
              <td className="num">{station.n}</td>
              <td className="num">{station.mean_discharge.toFixed(1)}</td>
              <td className={`num ${station.r2 < 0 ? "below" : ""}`}>
                {signed(station.r2)}
              </td>
              <td className="num">{signed(station.bias)}</td>
              <td>
                {station.in_training_range ? (
                  <span className="small muted">in range</span>
                ) : (
                  <span className="small flagged">extrapolating</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** R² and bias are signed quantities; an unsigned +0.336 reads as a magnitude. */
function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}
