/**
 * Application shell: masthead, routes, and the degraded-backend banner.
 *
 * Model metadata is fetched once here and passed down, so every page shows the
 * same provenance and the same training ranges.
 */

import { NavLink, Route, Routes } from "react-router-dom";

import About from "./pages/About";
import Analyze from "./pages/Analyze";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Scenarios from "./pages/Scenarios";
import { API_BASE } from "./services/api";
import { useModels } from "./useModels";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/predict", label: "Predict" },
  { to: "/scenarios", label: "Scenarios" },
  { to: "/analyze", label: "Analyze" },
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
              AquaNexus <span>· 綾瀬川</span>
            </NavLink>
            <span className="small" style={{ color: "#9fc3d8" }}>
              {models.health
                ? `API ${models.health.status} · v${models.health.version}`
                : "connecting…"}
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
              <code>uvicorn aquanexus.api.app:app --reload</code>, and if it
              reports no models, run <code>python scripts/train_models.py</code>{" "}
              first.{" "}
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

        {models.loading && !models.error && (
          <p className="spinner">Loading model metadata…</p>
        )}

        <Routes>
          <Route path="/" element={<Home models={models} />} />
          <Route path="/predict" element={<Predict models={models} />} />
          <Route path="/scenarios" element={<Scenarios models={models} />} />
          <Route path="/analyze" element={<Analyze models={models} />} />
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
          Data: 埼玉県 河川点群データ (CC BY 4.0) · 環境省 / 埼玉県
          公共用水域水質測定 · 気象庁. Predictions are a proof of concept, not
          regulatory advice — the habitat index is trained on synthetic labels and
          the oxygen model is unreliable at low flow.
        </div>
      </footer>
    </div>
  );
}
