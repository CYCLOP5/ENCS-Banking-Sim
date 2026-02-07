/**
 * Eager topology cache — starts fetching the moment this module is imported.
 * Import it from App.jsx so the fetch fires on first page load,
 * then consume the same promise in Simulation.jsx.
 */
import { fetchTopology } from "./api";

// Kick off the request immediately at module-load time
let _promise = null;

export function preloadTopology() {
  if (!_promise) {
    _promise = fetchTopology().catch((err) => {
      // Allow retry on next call
      _promise = null;
      throw err;
    });
  }
  return _promise;
}

// Fire it right now so it starts while the user is on the landing page
preloadTopology();
