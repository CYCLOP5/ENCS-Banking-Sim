/**
 * API service layer — connects the React frontend to the ENCS FastAPI backend.
 */

const BASE = "/api";

async function request(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API error");
  }
  return res.json();
}

/* ── Health ─────────────────────────────────────────────────────── */

export function fetchHealth() {
  return request("/health");
}

/* ── Topology (for 3D graph) ────────────────────────────────────── */

export function fetchTopology() {
  return request("/topology");
}

/* ── Bank data (for explorer table) ─────────────────────────────── */

export function fetchBanks() {
  return request("/banks");
}

/* ── Simulation (Eisenberg-Noe / Intraday) ──────────────────────── */

export function runSimulation(params = {}) {
  return request("/simulate", {
    method: "POST",
    body: JSON.stringify({
      trigger_idx: params.triggerIdx ?? 0,
      severity: params.severity ?? 1.0,
      max_iter: params.maxIter ?? 100,
      tolerance: params.tolerance ?? 1e-5,
      distress_threshold: params.distressThreshold ?? 0.95,
      use_intraday: params.useIntraday ?? true,
      n_steps: params.nSteps ?? 10,
      sigma: params.sigma ?? 0.05,
      panic_rate: params.panicRate ?? 0.1,
      fire_sale_alpha: params.fireSaleAlpha ?? 0.005,
      margin_multiplier: params.marginMultiplier ?? 1.0,
      use_ccp: params.useCcp ?? false,
      clearing_rate: params.clearingRate ?? 0.5,
      default_fund_ratio: params.defaultFundRatio ?? 0.05,
    }),
  });
}

/* ── Climate (Green Swan) ───────────────────────────────────────── */

export function runClimate(params = {}) {
  return request("/climate", {
    method: "POST",
    body: JSON.stringify({
      carbon_tax: params.carbonTax ?? 0.5,
      green_subsidy: params.greenSubsidy ?? 0.1,
      use_intraday: params.useIntraday ?? true,
      trigger_idx: params.triggerIdx ?? 0,
      severity: params.severity ?? 1.0,
      n_steps: params.nSteps ?? 10,
    }),
  });
}

/* ── Game Theory (Morris & Shin) ────────────────────────────────── */

export function runGame(params = {}) {
  return request("/game", {
    method: "POST",
    body: JSON.stringify({
      n_banks: params.nBanks ?? 20,
      n_steps: params.nSteps ?? 5,
      true_solvency: params.trueSolvency ?? 0.2,
      interest_rate: params.interestRate ?? 0.1,
      recovery_rate: params.recoveryRate ?? 0.4,
      risk_aversion: params.riskAversion ?? 1.0,
      noise_std: params.noiseStd ?? 0.08,
      haircut: params.haircut ?? 0.2,
      margin_pressure: params.marginPressure ?? 0.3,
      exposure: params.exposure ?? 1e9,
    }),
  });
}

/* ── GNN Risk Scores ────────────────────────────────────────────── */

export function fetchGnnRisk() {
  return request("/gnn-risk");
}
