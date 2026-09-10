# AquaNexus frontend

React + TypeScript dashboard over the AquaNexus API. Vite for the build, plain
CSS for the styling, inline SVG for the charts.

```bash
npm install
npm run dev      # http://localhost:3000, expects the API on :8000
npm run test     # 33 tests
npm run build    # typecheck + production bundle into dist/
```

Point it somewhere else with `VITE_API_BASE_URL` in `.env.local`. That is the
local knob and it wins when set. In a container the address instead comes from
`$API_BASE_URL`, which the entrypoint writes into `public/config.js` at start-up
— the page reads it at load, so one built image can serve any environment.

## What it has to get right

One of the two models served is trained on labels this project generated. A
dashboard that renders a habitat score as a confident number, with the caveat
one click away, would undo the thing the rest of the repository is careful
about. So:

- every prediction renders its provenance badge in the same card, never behind a
  disclosure;
- the caveats the API returns are shown with the number, not summarised;
- the collinearity warning is rendered *above* the SHAP chart, because the bar
  order is not a ranking and a reader who sees the chart first has already
  concluded that it is;
- the training range comes from `/models` rather than being copied into the
  frontend, so it cannot drift when the models are retrained;
- an input outside that range is allowed and flagged, matching what the API
  does — refusing the question would hide that the model has an edge.

`src/test/provenance.test.tsx` exists to keep those properties from being
refactored away.

## Layout

```
src/
├── App.tsx                  shell, routes, degraded-backend banner
├── useModels.ts             /health + /models, fetched once and shared
├── services/api.ts          typed client (fetch, no axios)
├── types.ts                 mirrors aquanexus.api.schemas
├── components/
│   ├── Provenance.tsx       badge, caveat list, error banner
│   ├── EnvironmentalInput.tsx
│   ├── PredictionCard.tsx
│   ├── ExplainabilityPanel.tsx   SHAP bars, inline SVG
│   ├── InteractionHeatmap.tsx    response surface, inline SVG
│   ├── TransferScale.tsx         held-out river against its datums, inline SVG
│   ├── ScenarioBuilder.tsx
│   └── TargetPicker.tsx
└── pages/                   Home, Predict, Scenarios, Analyze, Transfer, About
```

## Choices worth explaining

**`fetch`, not axios** (`AQUANEXUS-PLAN.md` §4a specifies axios). The client is
about a hundred lines of request building; a dependency there would need
mocking in every component test for no benefit.

**Hand-written CSS, not Material-UI or Tailwind.** A dozen components do not
need a framework, and the tokens in `styles.css` are the same ink, accent and
cool blue that `scripts/make_figures.py` draws the README figures with — so the
app and the plots read as one project.

**Inline SVG, not a chart library.** Two chart types, both simple, both needing
custom annotation (the collinearity marks, the out-of-range colouring). A charting
dependency would be larger than the charts.
