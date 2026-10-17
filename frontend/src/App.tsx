/**
 * Application shell: masthead, routes, and the degraded-backend banner.
 *
 * Model metadata is fetched once here and passed down, so every page shows the
 * same held-out accuracy.
 */

import { NavLink, Route, Routes } from "react-router-dom";

import About from "./pages/About";
import Home from "./pages/Home";
import Models from "./pages/Models";
import Predict from "./pages/Predict";
import { API_BASE } from "./services/api";
import { useModels } from "./useModels";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/predict", label: "Predict" },
  { to: "/models", label: "Models" },
  { to: "/about", label: "About" },
];

export default function App() {
  const models = useModels();

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-inner">
          <div className="masthead-top">
            <NavLink to="/" className="wordmark">
              <b>AquaNexus</b>
              <span>Dissolved oxygen in Japanese rivers</span>
            </NavLink>
            <span className={`station ${models.health?.status === "ok" ? "" : "down"}`}>
              <i aria-hidden="true" />
              {models.health
                ? models.health.status === "ok"
                  ? `${models.health.models_loaded.length} models, ${models.health.stations.toLocaleString()} stations`
                  : "running without models"
                : "connecting to the API"}
            </span>
          </div>
          <nav className="nav">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? "active" : "")}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="page">
        {models.error && (
          <div className="banner banner-error" role="alert">
            <strong>The API is not answering with models.</strong> {models.error}
            <div className="small" style={{ marginTop: "0.35rem" }}>
              Expected at <code>{API_BASE}</code>. Start it with{" "}
              <code>uvicorn aquanexus.api.app:app --reload</code>, and if it reports no
              models, run <code>python scripts/save_do_model.py</code> first.{" "}
              <button
                type="button"
                className="secondary"
                onClick={models.reload}
                style={{ marginLeft: "0.4rem", padding: "0.2rem 0.6rem" }}
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {models.loading && !models.error && <p className="spinner">Loading model metadata…</p>}

        <Routes>
          <Route path="/" element={<Home models={models} />} />
          <Route path="/predict" element={<Predict models={models} />} />
          <Route path="/models" element={<Models models={models} />} />
          <Route path="/about" element={<About />} />
          <Route
            path="*"
            element={
              <section className="card">
                <h1>Not found</h1>
                <p className="muted">No such page.</p>
              </section>
            }
          />
        </Routes>
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <p>
            Water quality from 環境省 公共用水域水質測定 (FY2013-22). Weather from NASA POWER.
            Elevation from 国土地理院. Proof of concept, not regulatory advice.
          </p>
        </div>
      </footer>
    </div>
  );
}
