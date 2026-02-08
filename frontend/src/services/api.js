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

const _cache = {
  topology: null,
  banks: null,
};

/* ── Health ─────────────────────────────────────────────────────── */

export function fetchHealth() {
  return request("/health");
}

/* ── Topology (for 3D graph) ────────────────────────────────────── */

export async function fetchTopology(forceRefresh = false) {
  if (_cache.topology && !forceRefresh) return _cache.topology;
  const data = await request("/topology");
  _cache.topology = data;
  return data;
}

/* ── Bank data (for explorer table) ─────────────────────────────── */

export async function fetchBanks(forceRefresh = false) {
  if (_cache.banks && !forceRefresh) return _cache.banks;
  const data = await request("/banks");
  _cache.banks = data;
  return data;
}

/* ── Simulation (Eisenberg-Noe / Intraday) ──────────────────────── */

export function runSimulation(params = {}) {
  return request("/simulate", {
    method: "POST",
    body: JSON.stringify({
      topology_type: params.topologyType ?? "smart",
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
      circuit_breaker_enabled: params.circuitBreakerEnabled ?? false,
      circuit_breaker_threshold: params.circuitBreakerThreshold ?? 0.15,
      // Strategic
      use_strategic: params.useStrategic ?? false,
      strategic_alpha: params.strategicAlpha ?? 5.0,
      strategic_risk_aversion: params.strategicRiskAversion ?? 1.0,
      strategic_interest_rate: params.strategicInterestRate ?? 0.05,
      strategic_recovery_rate: params.strategicRecoveryRate ?? 0.40,
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
      circuit_breaker_enabled: params.circuitBreakerEnabled ?? false,
      circuit_breaker_threshold: params.circuitBreakerThreshold ?? 0.15,
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

/* ── LLM Explain ───────────────────────────────────── */

export function explainRun(params = {}) {
  return request("/explain/run", {
    method: "POST",
    body: JSON.stringify({
      run_id: params.runId ?? null,
      run_type: params.runType ?? null,
      question: params.question ?? "Summarize the simulation results.",
    }),
  });
}

export function explainBank(params = {}) {
  return request("/explain/bank", {
    method: "POST",
    body: JSON.stringify({
      bank_id: params.bankId,
      run_id: params.runId ?? null,
      question: params.question ?? "Explain this bank's risk profile.",
    }),
  });
}

/* ── Chatbot ───────────────────────────────────────── */

export function chatWithAssistant(params = {}) {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({
      messages: params.messages ?? [],
      run_id: params.runId ?? null,
      run_type: params.runType ?? null,
      bank_id: params.bankId ?? null,
      bank_name: params.bankName ?? null,
    }),
  });
}
