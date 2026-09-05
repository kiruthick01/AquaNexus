# AquaNexus

**Physics-informed ML framework for aquatic ecosystem diagnosis.**

AquaNexus couples HEC-RAS hydraulic simulation with gradient-boosted regression to
predict a Habitat Suitability Index (HSI, 0–1) from environmental state, and uses
SHAP to explain *why* a given reach scores the way it does.

> **Status:** early development. Phase 1a (scaffold) complete; data pipeline in progress.

---

## Scope and honest caveats

Read this before interpreting any metric this project reports.

**1. The training labels are synthetic.** HSI is not measured in the field here. It is
generated from published ecological response curves over temperature, dissolved oxygen,
depth, velocity and sediment (`docs/ML_METHODOLOGY.md`). Because the label is computed
from the same variables that become model features, a high R² demonstrates that the model
recovered a known function — **not** that it predicts real habitat suitability. The
linear-regression baseline underperforming XGBoost shows the label function is nonlinear,
which is by construction.

What the project does legitimately demonstrate: a working HEC-RAS→ML→API→UI pipeline,
correct validation methodology, and explainability tooling. Treat it as an engineering
and methodology artefact, not an ecological finding.

**2. Real observations anchor the ranges, not the labels.** Monthly water-quality records
from the Japan Ministry of the Environment bound the plausible feature space and validate
that simulated conditions are realistic. They are not used as ground-truth HSI.

**3. The study site changed.** The original plan targeted the Yodo River. An availability
audit (`docs/DATA_SOURCES.md`, 2026-09-06) found no open channel bathymetry for it. The
project now targets the **Ayase (綾瀬川) / Naka (中川) rivers in Saitama**, which have
CC BY 4.0 point-cloud data acquired by drone *and narrow multibeam echosounder* — the only
open Japanese source found that supplies below-water channel geometry.

---

## Data sources

| Purpose | Source | License |
|---|---|---|
| Channel geometry | [埼玉県 河川点群データ](https://www.geospatial.jp/ckan/dataset/river-pointcloud-saitama) | CC BY 4.0 |
| Water quality | [環境省 水環境総合情報サイト](https://water-pub.env.go.jp/water-pub/mizu-site/mizu/download/download.asp) | Open |
| Meteorology | [気象庁 過去の気象データ](https://www.data.jma.go.jp/risk/obsdl/) | 公共データ利用規約 1.0 |
| Discharge / stage | [MLIT 水文水質データベース](https://www1.river.go.jp/) | Manual download only |

> ⚠️ **Do not scrape `river.go.jp`.** It explicitly prohibits automated acquisition.
> Discharge data is collected by hand through its web UI, 30 days per request.

Full audit with rejected alternatives: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

---

## Quickstart

```bash
git clone https://github.com/kiruthick01/aquanexus.git
cd aquanexus

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[ml,api,dev]"   # add ",geo" for point-cloud processing
cp .env.example .env             # then edit HECRAS_EXE to match your install

pytest
```

Serve the API (once Phase 3 lands):

```bash
uvicorn aquanexus.api.app:app --reload
# docs at http://localhost:8000/docs
```

Or with Docker:

```bash
docker compose up --build
```

---

## Layout

```
src/aquanexus/
  config.py     settings singleton (pydantic-settings)
  logger.py     logging setup
  hecras/       read, parse, run and validate HEC-RAS projects
  data/         loading, preprocessing, quality checks, synthetic labels
  ml/           models, training, SHAP explainer, evaluation, inference
  api/          FastAPI app, routes, schemas, middleware
  dashboard/    Streamlit demo
  utils/        I/O, metrics, plotting helpers
frontend/       React + TypeScript dashboard
notebooks/      exploration and training notebooks
scripts/        CLI entry points for the pipeline stages
docs/           architecture, data sources, methodology, API reference
```

## Requirements

- Python 3.11+
- HEC-RAS 6.x (Windows only) — required to *generate* simulation data, not to run the API
- Node 18+ for the frontend

## License

MIT. Data retains its original licensing; attribute the sources listed above.
