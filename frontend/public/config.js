/*
 * Runtime configuration, loaded before the bundle.
 *
 * Vite substitutes import.meta.env when the bundle is BUILT, so a built image
 * baked to one API address could never be repointed without rebuilding. This
 * file is read at page load instead: the container's entrypoint rewrites it
 * from $API_BASE_URL at start-up, and one image serves any environment.
 *
 * In development this file is served as-is and VITE_API_BASE_URL in .env.local
 * still wins, because that is the more convenient knob when running locally.
 */
window.__AQUANEXUS_CONFIG__ = {
  apiBaseUrl: "",
};
