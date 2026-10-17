# AquaNexus frontend

React + TypeScript dashboard over the AquaNexus API. Vite for the build, plain CSS
for the styling.

```bash
npm install
npm run dev      # http://localhost:3000, expects the API on :8000
npm run test
npm run build    # typecheck + production bundle into dist/
```

Point it somewhere else with `VITE_API_BASE_URL` in `.env.local`. That is the local
knob and it wins when set. In a container the address instead comes from
`$API_BASE_URL`, which the entrypoint writes into `public/config.js` at start-up; the
page reads it at load, so one built image can serve any environment.

## Pages

- **Overview**: held-out accuracy read from `GET /models`, never hard-coded.
- **Predict**: a place (station search or coordinates), a date and weather in; DO, a 90%
  interval, the inputs that were missing, and the biggest contributions out.
- **Models**: each served model with its held-out R², error, low-oxygen recall and
  precision, feature importance and interval widths.
- **About**: data sources, method and limits.

## What it has to get right

The honest numbers are the product. Accuracy shown here comes from the API, which takes
it from the evaluation on unseen stations. A prediction always shows its interval, the
inputs it did not have, and any value outside the training range beside the number.
