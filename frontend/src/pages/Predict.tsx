/**
 * Predict: a place, a date and some weather in; oxygen, an interval and the reasons out.
 */

import { useState } from "react";

import { api, ApiError } from "../services/api";
import type { Mode, PredictionResponse, StationInfo, Weather } from "../types";
import type { ModelState } from "../useModels";

const WEATHER_FIELDS: { key: keyof Weather; label: string; unit: string }[] = [
  { key: "air_temp", label: "Air temperature", unit: "°C" },
  { key: "air_temp_3d", label: "Mean air temperature, previous 3 days", unit: "°C" },
  { key: "air_temp_7d", label: "Mean air temperature, previous 7 days", unit: "°C" },
  { key: "air_temp_30d", label: "Mean air temperature, previous 30 days", unit: "°C" },
  { key: "rain_7d", label: "Rain in the previous 7 days", unit: "mm" },
  { key: "rain_30d", label: "Rain in the previous 30 days", unit: "mm" },
];

const today = () => new Date().toISOString().slice(0, 10);

export default function Predict({ models }: { models: ModelState }) {
  const [mode, setMode] = useState<Mode>("geo");
  const [date, setDate] = useState(today());
  const [query, setQuery] = useState("");
  const [found, setFound] = useState<StationInfo[]>([]);
  const [station, setStation] = useState<StationInfo | null>(null);
  const [latitude, setLatitude] = useState("35.8");
  const [longitude, setLongitude] = useState("139.8");
  const [weather, setWeather] = useState<Weather>({ air_temp: 25 });
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fail = (caught: unknown) =>
    setError(caught instanceof ApiError ? caught.message : String(caught));

  const search = async () => {
    setError(null);
    try {
      setFound(await api.stations(query));
    } catch (caught) {
      fail(caught);
    }
  };

  const predict = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.predict({
          date,
          mode,
          explain: true,
          weather,
          ...(station
            ? { station_id: station.station_id }
            : { location: { latitude: Number(latitude), longitude: Number(longitude) } }),
        }),
      );
    } catch (caught) {
      fail(caught);
    } finally {
      setBusy(false);
    }
  };

  const setNumber = (key: keyof Weather, raw: string) =>
    setWeather((current) => {
      const next = { ...current };
      if (raw === "") delete next[key];
      else next[key] = Number(raw);
      return next;
    });

  const needsHistory = mode === "full" || mode === "full_alert";
  const contributions = result?.contributions
    ? Object.entries(result.contributions)
        .filter(([name]) => name !== "bias")
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 8)
    : [];

  return (
    <>
      <h1>Predict</h1>
      <p className="muted prose">
        Pick a place and a date, add whatever weather you know, and choose how much the model may
        assume. Anything left blank is treated as unknown and listed in the result.
      </p>

      {error && (
        <div className="banner banner-error" role="alert">
          {error}
        </div>
      )}

      <div className="split">
        <section className="card" aria-label="Inputs">
          <div className="field">
            <label htmlFor="mode">Model</label>
            <select id="mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              {(models.models.length ? models.models.map((m) => m.mode) : ["geo"]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <div className="small muted">{models.byMode(mode)?.summary}</div>
          </div>

          <div className="field">
            <label htmlFor="date">Date</label>
            <input id="date" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
          </div>

          <div className="field">
            <label htmlFor="station-search">Station (optional)</label>
            <div className="button-row">
              <input
                id="station-search"
                placeholder="station or river name, e.g. 綾瀬川"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <button type="button" className="secondary" onClick={search}>
                Search
              </button>
            </div>
            {found.length > 0 && !station && (
              <ul className="small" style={{ margin: "0.4rem 0 0", paddingLeft: "1.1rem" }}>
                {found.map((s) => (
                  <li key={s.station_id}>
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => setStation(s)}
                      style={{ padding: "0.1rem 0.5rem" }}
                    >
                      {s.name ?? s.station_id} · {s.water_body}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {station && (
              <div className="small">
                Using <strong>{station.name}</strong> ({station.water_body}); its location and
                terrain are filled in.{" "}
                <button
                  type="button"
                  className="secondary"
                  onClick={() => setStation(null)}
                  style={{ padding: "0.1rem 0.5rem" }}
                >
                  Clear
                </button>
              </div>
            )}
          </div>

          {!station && (
            <div className="pair">
              <div className="field">
                <label htmlFor="lat">Latitude</label>
                <input id="lat" value={latitude} onChange={(e) => setLatitude(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="lon">Longitude</label>
                <input id="lon" value={longitude} onChange={(e) => setLongitude(e.target.value)} />
              </div>
            </div>
          )}

          {WEATHER_FIELDS.map(({ key, label, unit }) => (
            <div className="field" key={key}>
              <label htmlFor={key}>
                {label} ({unit})
              </label>
              <input
                id={key}
                type="number"
                value={weather[key] ?? ""}
                onChange={(e) => setNumber(key, e.target.value)}
              />
            </div>
          ))}

          {needsHistory && !station && (
            <p className="small muted">
              The history models need a station: without one their history inputs are missing and
              they fall back to roughly the weather-only accuracy.
            </p>
          )}

          <div className="button-row">
            <button type="button" onClick={predict} disabled={busy}>
              {busy ? "Predicting…" : "Predict"}
            </button>
          </div>
        </section>

        <div>
          {result ? (
            <section className="card" aria-label="Result">
              <div className="reading">
                <span className="reading-value">{result.prediction.toFixed(2)}</span>
                <span className="reading-unit">mg/L</span>
              </div>
              <p className="interpretation">{result.interpretation}</p>
              <p>
                {Math.round(result.level * 100)}% interval:{" "}
                <strong>
                  {result.lower.toFixed(2)} to {result.upper.toFixed(2)} mg/L
                </strong>
              </p>
              {result.notes.map((note) => (
                <div className="banner" key={note}>
                  {note}
                </div>
              ))}
              {result.out_of_range.length > 0 && (
                <div className="banner" role="status">
                  Outside the range the model saw in training:{" "}
                  {result.out_of_range.map((w) => w.feature).join(", ")}.
                </div>
              )}
              {contributions.length > 0 && (
                <>
                  <h3>What moved the prediction (mg/L)</h3>
                  <ul className="small mono" style={{ listStyle: "none", padding: 0 }}>
                    {contributions.map(([name, value]) => (
                      <li key={name}>
                        {name.padEnd(18, " ")} {value >= 0 ? "+" : ""}
                        {value.toFixed(2)}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {result.missing_features.length > 0 && (
                <p className="small muted">
                  Not supplied ({result.missing_features.length}): {result.missing_features.join(", ")}
                </p>
              )}
              <ul className="caveats small">
                {result.caveats.map((caveat) => (
                  <li key={caveat}>{caveat}</li>
                ))}
              </ul>
            </section>
          ) : (
            <section className="card">
              <p className="muted" style={{ margin: 0 }}>
                The defaults describe a warm day near the Ayase River. Change them, then press
                Predict.
              </p>
            </section>
          )}
        </div>
      </div>
    </>
  );
}
