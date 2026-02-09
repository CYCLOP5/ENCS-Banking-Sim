from __future__ import annotations
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import traceback
import os
import logging
import ctypes as _ctypes
import uuid
from dotenv import load_dotenv
import simulation_engine as sim
from strategic_model import run_game_simulation
from climate_risk import assign_climate_exposure, run_transition_shock
from pathlib import Path
load_dotenv()
from llm_store import LlmStore
from llm_explain import (
    build_run_summary,
    build_bank_context,
    build_prompt,
    build_bank_prompt,
    call_groq,
    build_graph_evidence,
    load_site_knowledge,
    call_groq_chat,
)
logger = logging.getLogger(__name__)
def _preload_itt_stub() -> str | None:
    candidates = [
        os.environ.get("ENCS_ITT_STUB_PATH"),
        os.environ.get("ITT_NOTIFY_STUB"),
        str(Path(__file__).parent / "libittnotify_stub.so"),
    ]
    for stub_path in candidates:
        if stub_path and os.path.isfile(stub_path):
            try:
                _ctypes.CDLL(stub_path, mode=getattr(_ctypes, "RTLD_GLOBAL", 0))
                return stub_path
            except Exception:
                continue
    return None
ML_AVAILABLE = False
ML_ERROR: str | None = None
_ITT_STUB_USED = None
try:
    _ITT_STUB_USED = _preload_itt_stub()
    import torch
    from ml_pipeline import load_trained_model, predict_risk
    ML_AVAILABLE = True
except Exception as exc:
    ML_AVAILABLE = False
    ML_ERROR = f"{type(exc).__name__}: {exc}"
GNN_MODEL_PATH = Path(__file__).parent / "gnn_model.pth"
app = FastAPI(
    title="ENCS Systemic Risk Engine API",
    version="1.0.0",
    description="REST API for the Eisenberg-Noe Contagion Simulation engine.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_cache: Dict[str, Any] = {}
_llm_store = LlmStore()
def _get_cached_model():
    if "gnn_model" in _cache:
        return _cache["gnn_model"]
    if not ML_AVAILABLE:
        return None
    if not GNN_MODEL_PATH.exists():
        _cache["gnn_model_error"] = f"Model not found at {GNN_MODEL_PATH}"
        return None
    try:
        model = load_trained_model(str(GNN_MODEL_PATH))
        _cache["gnn_model"] = model
        return model
    except Exception as exc:
        _cache["gnn_model_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to load GNN model")
        return None
def _load_network():
    """Load network once and cache."""
    if "W_dense" not in _cache:
        W_sparse, df = sim.load_and_align_network()
        W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)
        _cache["W_sparse"] = W_sparse
        _cache["W_dense"] = W_dense
        _cache["df"] = df
    return _cache["W_dense"], _cache["df"]
def _generate_max_entropy_topology(df: pd.DataFrame, target_total_weight: float) -> np.ndarray:
    """Generates a Max Entropy (Uniform) topology with the same total volume."""
    n = len(df)
    if n <= 1:
        return np.zeros((n, n))
    num_edges = n * (n - 1)
    if target_total_weight <= 0:
        return np.zeros((n, n))
    avg_weight = target_total_weight / num_edges
    W_uniform = np.full((n, n), avg_weight)
    np.fill_diagonal(W_uniform, 0.0)
    return W_uniform
class SimulationRequest(BaseModel):
    topology_type: str = Field("smart", pattern="^(smart|uniform)$")
    trigger_idx: int = 0
    severity: float = Field(1.0, ge=0, le=1)
    max_iter: int = Field(100, ge=10, le=500)
    tolerance: float = 1e-5
    distress_threshold: float = Field(0.95, ge=0, le=1)
    use_intraday: bool = True
    n_steps: int = Field(10, ge=1, le=50)
    sigma: float = Field(0.05, ge=0.01, le=0.30)
    panic_rate: float = Field(0.10, ge=0, le=0.50)
    fire_sale_alpha: float = Field(0.005, ge=0, le=0.05)
    margin_multiplier: float = Field(1.0, ge=0, le=5)
    use_strategic: bool = False
    strategic_interest_rate: float = Field(0.05, ge=0.01, le=0.20)
    strategic_recovery_rate: float = Field(0.40, ge=0.10, le=0.80)
    strategic_risk_aversion: float = Field(1.0, ge=0.1, le=10.0)
    strategic_info_regime: str = Field("OPAQUE")  
    strategic_alpha: float = Field(5.0, ge=0.01, le=100.0)  
    strategic_noise_std: float = Field(0.05, ge=0.01, le=2.0)
    strategic_haircut: float = Field(0.20, ge=0.01, le=1.0)
    strategic_margin_pressure: float = Field(0.5, ge=0.0, le=5.0)
    strategic_exposure_scale: float = Field(1.0, ge=0.1, le=100.0) 
    circuit_breaker_enabled: bool = False
    circuit_breaker_threshold: float = Field(0.15, ge=0.01, le=0.50)
    use_ccp: bool = False
    clearing_rate: float = Field(0.5, ge=0, le=1)
    default_fund_ratio: float = Field(0.05, ge=0.01, le=0.25)
class ClimateRequest(BaseModel):
    carbon_tax: float = Field(0.5, ge=0, le=1)
    green_subsidy: float = Field(0.10, ge=0, le=1.0)
    use_intraday: bool = True
    trigger_idx: int = 0
    severity: float = Field(1.0, ge=0, le=1)
    n_steps: int = Field(10, ge=1, le=50)
    circuit_breaker_enabled: bool = False
    circuit_breaker_threshold: float = Field(0.15, ge=0.01, le=0.50)
class GameRequest(BaseModel):
    n_banks: int = Field(20, ge=5, le=100)
    n_steps: int = Field(5, ge=2, le=20)
    true_solvency: float = Field(0.20, ge=-0.05, le=0.30)
    interest_rate: float = Field(0.10, ge=0.01, le=0.20)
    recovery_rate: float = Field(0.40, ge=0.10, le=0.80)
    risk_aversion: float = Field(1.0, ge=0.1, le=3.0)
    noise_std: float = Field(0.08, ge=0.01, le=0.30)
    haircut: float = Field(0.20, ge=0.05, le=0.50)
    margin_pressure: float = Field(0.30, ge=0, le=1)
    exposure: float = Field(1e9, ge=1e8, le=5e10)
class ExplainRunRequest(BaseModel):
    run_id: Optional[str] = None
    run_type: Optional[str] = None
    question: Optional[str] = "Summarize the simulation results in simple terms."
class ExplainBankRequest(BaseModel):
    bank_id: str
    run_id: Optional[str] = None
    question: Optional[str] = "Explain this bank's risk profile in simple terms."
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    run_id: Optional[str] = None
    run_type: Optional[str] = None
    bank_id: Optional[str] = None
    bank_name: Optional[str] = None
def _ndarray_to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
def _clean_results(d: dict) -> dict:
    """Convert numpy arrays and other non-serializable types to JSON-safe."""
    import math
    def _sanitize_float(v):
        """Replace NaN/Inf with None (JSON null)."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    def _sanitize_list(lst):
        return [
            _sanitize_float(x) if isinstance(x, (int, float)) else x
            for x in lst
        ]
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            v_safe = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            out[k] = v_safe.tolist()
        elif isinstance(v, dict):
            out[k] = _clean_results(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v) if np.isfinite(v) else 0.0
        elif isinstance(v, list):
            cleaned_list = []
            for item in v:
                if isinstance(item, np.ndarray):
                    cleaned_list.append(
                        np.nan_to_num(item, nan=0.0, posinf=0.0, neginf=0.0).tolist()
                    )
                elif isinstance(item, (np.floating, float)):
                    cleaned_list.append(float(item) if np.isfinite(item) else 0.0)
                elif isinstance(item, (np.integer, int)):
                    cleaned_list.append(int(item))
                elif isinstance(item, dict):
                    cleaned_list.append(_clean_results(item))
                else:
                    cleaned_list.append(item)
            out[k] = cleaned_list
        elif k == "agents":
            continue
        else:
            out[k] = v
    return out
def _build_bank_snapshot(df: pd.DataFrame) -> list[dict]:
    snapshot = []
    for i, row in df.iterrows():
        snapshot.append({
            "id": int(i),
            "bank_id": str(row.get("bank_id", "")),
            "name": str(row.get("bank_name", "")),
            "region": str(row.get("region", "")),
            "tier": str(row.get("tier", "")),
            "total_assets": float(row.get("total_assets", 0) or 0),
            "total_liabilities": float(row.get("total_liabilities", 0) or 0),
            "equity": float(row.get("equity_capital", 0) or 0),
        })
    return snapshot
def _extract_top_losses(results: dict, top_n: int = 5) -> list[dict]:
    names = results.get("bank_names") or []
    initial = results.get("initial_equity") or []
    final = results.get("final_equity") or []
    rows = []
    for i, name in enumerate(names):
        if i < len(initial) and i < len(final):
            loss = (initial[i] or 0) - (final[i] or 0)
            rows.append({"name": str(name), "equity_loss": float(loss)})
    rows.sort(key=lambda r: r["equity_loss"], reverse=True)
    return rows[:top_n]
def _percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    below = sum(1 for v in values if v <= value)
    return round(100.0 * below / len(values), 2)
def _bank_peer_summary(all_banks: list[dict], bank: dict) -> dict:
    region = bank.get("region")
    tier = bank.get("tier")
    peers = [b for b in all_banks if b.get("region") == region and b.get("tier") == tier]
    return {
        "peer_group_size": len(peers),
        "region": region,
        "tier": tier,
    }
def _build_chat_evidence(req: ChatRequest) -> Dict[str, Any]:
    site_knowledge = load_site_knowledge()
    stored = (
        _llm_store.get_run(req.run_id)
        if req.run_id
        else _llm_store.get_latest_run(req.run_type)
    )
    run_summary = None
    graph_evidence = None
    if stored:
        run_summary = stored.summary_json or build_run_summary(stored.run_type, stored.result_json)
        graph_evidence = build_graph_evidence(stored.result_json)
    bank_profile = None
    if req.bank_id:
        bank_profile = _llm_store.get_bank_profile(req.bank_id)
    elif req.bank_name:
        bank_profile = _llm_store.find_bank_profile_by_name(req.bank_name)
    bank_context = None
    if bank_profile:
        bank_context = build_bank_context(
            bank_profile,
            stored.run_type if stored else None,
            stored.result_json if stored else None,
        )
    bank_count = _llm_store.get_bank_profile_count()
    include_all = os.environ.get("ENCS_CHAT_INCLUDE_ALL_BANKS", "0") == "1"
    top_banks = _llm_store.get_top_bank_profiles(limit=8)
    all_banks = None
    if include_all:
        all_banks = [
            {
                "bank_id": b.get("bank_id"),
                "name": b.get("name"),
                "region": b.get("region"),
                "tier": b.get("tier"),
                "total_assets": b.get("total_assets"),
                "equity": b.get("equity"),
                "leverage_ratio": b.get("leverage_ratio"),
                "gnn_risk_score": b.get("gnn_risk_score"),
                "carbon_score": b.get("carbon_score"),
            }
            for b in _llm_store.get_all_bank_profiles()
        ]
    return {
        "site": site_knowledge,
        "latest_run": {
            "run_id": stored.run_id if stored else None,
            "run_type": stored.run_type if stored else None,
            "summary": run_summary,
            "graphs": graph_evidence,
        },
        "bank_context": bank_context,
        "bank_profiles_sample": top_banks,
        "bank_profiles_count": bank_count,
        "bank_profiles_all": all_banks,
    }
@app.get("/api/health")
async def health():
    model_error = _cache.get("gnn_model_error")
    return {
        "status": "ok",
        "ml_available": ML_AVAILABLE,
        "ml_error": ML_ERROR or model_error,
        "itt_stub": _ITT_STUB_USED,
        "model_path": str(GNN_MODEL_PATH),
        "model_exists": GNN_MODEL_PATH.exists(),
    }
@app.get("/api/topology")
async def get_topology():
    """
    Return the network graph as JSON suitable for react-force-graph-3d.
    Nodes = banks, Links = interbank edges.
    """
    try:
        W_dense, df = _load_network()
        n = W_dense.shape[0]
        nodes = []
        for i, row in df.iterrows():
            nodes.append({
                "id": int(i),
                "bank_id": str(row.get("bank_id", "")),
                "name": str(row["bank_name"])[:40],
                "region": str(row["region"]),
                "tier": str(row.get("tier", "periphery")),
                "total_assets": float(row["total_assets"]),
                "equity": float(row.get("equity_capital", 0)),
                "leverage_ratio": float(row.get("leverage_ratio", 0)),
            })
        rows_idx, cols_idx = np.nonzero(W_dense)
        weights = W_dense[rows_idx, cols_idx]
        MAX_EDGES = 3000
        if len(weights) > MAX_EDGES:
            threshold = np.percentile(weights, 100 * (1 - MAX_EDGES / len(weights)))
            mask = weights >= threshold
        else:
            mask = np.ones(len(weights), dtype=bool)
        connected = set(rows_idx[mask]) | set(cols_idx[mask])
        for node_id in range(n):
            if node_id not in connected:
                out_mask = rows_idx == node_id
                in_mask = cols_idx == node_id
                node_mask = out_mask | in_mask
                if node_mask.any():
                    best = np.argmax(weights * node_mask)
                    mask[best] = True
        rows_idx, cols_idx, weights = rows_idx[mask], cols_idx[mask], weights[mask]
        links = []
        for s, t, w in zip(rows_idx, cols_idx, weights):
            links.append({
                "source": int(s),
                "target": int(t),
                "value": float(w),
            })
        return {"nodes": nodes, "links": links, "n_banks": len(df)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/banks")
async def get_banks():
    """Return full bank table for the Bank Explorer page."""
    try:
        W_dense, df = _load_network()
        if "climate_df_enriched" not in _cache:
            df_enr = df.copy()
            assign_climate_exposure(df_enr)
            _cache["climate_df_enriched"] = df_enr
        df_enriched = _cache["climate_df_enriched"].copy()
        risk_scores = np.zeros(len(df_enriched))
        if ML_AVAILABLE:
            model = _get_cached_model()
            if model is not None:
                try:
                    result = predict_risk(model, W_dense, df_enriched)
                    risk_scores = result["risk_scores"]
                except Exception:
                    logger.exception("Risk prediction failed")
        banks = []
        for i, row in df_enriched.iterrows():
            banks.append({
                "id": int(i),
                "bank_id": str(row.get("bank_id", "")),
                "name": str(row["bank_name"]),
                "region": str(row["region"]),
                "tier": str(row.get("tier", "periphery")),
                "total_assets": float(row["total_assets"]),
                "total_liabilities": float(row.get("total_liabilities", 0)),
                "equity": float(row.get("equity_capital", 0)),
                "leverage_ratio": float(row.get("leverage_ratio", 0)),
                "carbon_score": float(row.get("carbon_score", 0)),
                "brown_assets": float(row.get("brown_assets", 0)),
                "green_assets": float(row.get("green_assets", 0)),
                "gnn_risk_score": float(risk_scores[i]) if i < len(risk_scores) else 0.0,
            })
        if os.environ.get("ENCS_LLM_STORE_BANKS", "0") == "1":
            try:
                _llm_store.upsert_bank_profiles(banks)
            except Exception:
                logger.exception("Failed to store bank profiles")
        return {"banks": banks, "total": len(banks)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/simulate")
async def run_simulation(req: SimulationRequest):
    """Run the Eisenberg-Noe / Intraday simulation."""
    try:
        W_dense, df = _load_network()
        if req.use_strategic and req.strategic_exposure_scale != 1.0:
            current_avg_exposure = W_dense.sum(axis=1).mean() 
            if current_avg_exposure > 0:
                target_exposure = req.strategic_exposure_scale * 1e9 
                scale_factor = target_exposure / current_avg_exposure
                W_dense *= scale_factor
        if req.topology_type == "uniform":
            total_volume = W_dense.sum()
            W_dense = _generate_max_entropy_topology(df, total_volume)
        if req.use_ccp:
            W_dense, df = sim.apply_central_clearing(
                W_dense, df,
                clearing_rate=req.clearing_rate,
                default_fund_ratio=req.default_fund_ratio,
            )
        state = sim.compute_state_variables(W_dense, df)
        if req.use_strategic:
            opaque_result = sim.run_strategic_intraday_simulation(
                state, df.copy(),
                trigger_idx=req.trigger_idx,
                loss_severity=req.severity,
                n_steps=req.n_steps,
                uncertainty_sigma=req.strategic_noise_std, 
                alpha=req.strategic_haircut,               
                margin_sensitivity=req.strategic_margin_pressure,
                max_iterations=req.max_iter,
                convergence_threshold=req.tolerance,
                distress_threshold=req.distress_threshold,
                circuit_breaker_threshold=req.circuit_breaker_threshold if req.circuit_breaker_enabled else 0.0,
                interest_rate=req.strategic_interest_rate,
                recovery_rate=req.strategic_recovery_rate,
                risk_aversion_mean=req.strategic_risk_aversion,
                info_regime="OPAQUE",
                public_precision=None  
            )
            opaque_result["bank_names"] = df["bank_name"].tolist()
            transparent_result = sim.run_strategic_intraday_simulation(
                state, df.copy(),
                trigger_idx=req.trigger_idx,
                loss_severity=req.severity,
                n_steps=req.n_steps,
                uncertainty_sigma=req.strategic_noise_std, 
                alpha=req.strategic_haircut,               
                margin_sensitivity=req.strategic_margin_pressure,
                max_iterations=req.max_iter,
                convergence_threshold=req.tolerance,
                distress_threshold=req.distress_threshold,
                circuit_breaker_threshold=req.circuit_breaker_threshold if req.circuit_breaker_enabled else 0.0,
                interest_rate=req.strategic_interest_rate,
                recovery_rate=req.strategic_recovery_rate,
                risk_aversion_mean=req.strategic_risk_aversion,
                info_regime="TRANSPARENT",
                public_precision=req.strategic_alpha
            )
            transparent_result["bank_names"] = df["bank_name"].tolist()
            capital_saved = opaque_result.get('equity_loss', 0.0) - transparent_result.get('equity_loss', 0.0)
            run_id = str(uuid.uuid4())
            result_data = {
                "opaque": _clean_results(opaque_result),
                "transparent": _clean_results(transparent_result),
                "capital_saved": float(capital_saved),
                "run_id": run_id
            }
            try:
                summary = build_run_summary("game", result_data)
                _llm_store.save_run(
                    run_id=run_id,
                    run_type="game",
                    request_json=req.model_dump(),
                    result_json=result_data,
                    bank_snapshot_json={"banks": _build_bank_snapshot(df)},
                    summary_json=summary,
                )
            except Exception:
                logger.exception("Failed to store strategic run")
            return result_data
        if req.use_intraday:
            results = sim.run_rust_intraday(
                state, df,
                trigger_idx=req.trigger_idx,
                loss_severity=req.severity,
                n_steps=req.n_steps,
                uncertainty_sigma=req.sigma,
                panic_threshold=req.panic_rate,
                alpha=req.fire_sale_alpha,
                margin_sensitivity=req.margin_multiplier,
                max_iterations=req.max_iter,
                convergence_threshold=req.tolerance,
                distress_threshold=req.distress_threshold,
                circuit_breaker_threshold=req.circuit_breaker_threshold if req.circuit_breaker_enabled else 0.0,
            )
        else:
            results = sim.run_scenario(
                state, df,
                trigger_idx=req.trigger_idx,
                loss_severity=req.severity,
                max_iterations=req.max_iter,
                convergence_threshold=req.tolerance,
                distress_threshold=req.distress_threshold,
            )
        results["bank_names"] = df["bank_name"].tolist()
        cleaned = _clean_results(results)
        run_id = str(uuid.uuid4())
        try:
            summary = build_run_summary("mechanical", cleaned)
            _llm_store.save_run(
                run_id=run_id,
                run_type="mechanical",
                request_json=req.model_dump(),
                result_json=cleaned,
                bank_snapshot_json={"banks": _build_bank_snapshot(df)},
                summary_json=summary,
            )
        except Exception:
            logger.exception("Failed to store simulation run")
        cleaned["run_id"] = run_id
        return cleaned
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/climate")
async def run_climate(req: ClimateRequest):
    """Run the Green Swan transition shock."""
    try:
        W_dense, df = _load_network()
        df = df.copy()
        assign_climate_exposure(df)
        state = sim.compute_state_variables(W_dense, df)
        results = run_transition_shock(
            state, df,
            carbon_tax_severity=req.carbon_tax,
            green_subsidy=req.green_subsidy,
            use_intraday=req.use_intraday,
            n_steps=req.n_steps,
            circuit_breaker_threshold=req.circuit_breaker_threshold if req.circuit_breaker_enabled else 0.0,
        )
        results["bank_names"] = df["bank_name"].tolist()
        cleaned = _clean_results(results)
        run_id = str(uuid.uuid4())
        try:
            summary = build_run_summary("climate", cleaned)
            _llm_store.save_run(
                run_id=run_id,
                run_type="climate",
                request_json=req.model_dump(),
                result_json=cleaned,
                bank_snapshot_json={"banks": _build_bank_snapshot(df)},
                summary_json=summary,
            )
        except Exception:
            logger.exception("Failed to store climate run")
        cleaned["run_id"] = run_id
        return cleaned
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/game")
async def run_game(req: GameRequest):
    """Run Morris & Shin strategic game under both regimes."""
    try:
        common = dict(
            n_banks=req.n_banks,
            n_steps=req.n_steps,
            true_solvency=req.true_solvency,
            interest_rate=req.interest_rate,
            recovery_rate=req.recovery_rate,
            risk_aversion_mean=req.risk_aversion,
            private_noise_std=req.noise_std,
            fire_sale_haircut=req.haircut,
            margin_volatility=req.margin_pressure,
            initial_exposure_per_bank=req.exposure,
        )
        results_opaque = run_game_simulation(info_regime="OPAQUE", **common)
        results_transparent = run_game_simulation(info_regime="TRANSPARENT", **common)
        for res in (results_opaque, results_transparent):
            res["agent_names"] = [
                a.name for a in res.get("agents", [])
            ] or [f"Strategic Bank {i}" for i in range(req.n_banks)]
        cleaned = {
            "opaque": _clean_results(results_opaque),
            "transparent": _clean_results(results_transparent),
            "capital_saved": float(
                results_opaque.get("total_fire_sale_loss", 0)
                - results_transparent.get("total_fire_sale_loss", 0)
            ),
        }
        run_id = str(uuid.uuid4())
        try:
            summary = build_run_summary("game", cleaned)
            _llm_store.save_run(
                run_id=run_id,
                run_type="game",
                request_json=req.model_dump(),
                result_json=cleaned,
                bank_snapshot_json=None,
                summary_json=summary,
            )
        except Exception:
            logger.exception("Failed to store game run")
        cleaned["run_id"] = run_id
        return cleaned
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/gnn-risk")
async def get_gnn_risk():
    """Return GNN risk scores for all banks."""
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML dependencies not available")
    try:
        W_dense, df = _load_network()
        model = load_trained_model(str(GNN_MODEL_PATH))
        result = predict_risk(model, W_dense, df)
        scores = []
        for i, row in df.iterrows():
            scores.append({
                "id": int(i),
                "name": str(row["bank_name"]),
                "risk_score": float(result["risk_scores"][i]),
            })
        return {"scores": scores}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/explain/run")
async def explain_run(req: ExplainRunRequest):
    try:
        stored = (
            _llm_store.get_run(req.run_id)
            if req.run_id
            else _llm_store.get_latest_run(req.run_type)
        )
        if not stored:
            raise HTTPException(status_code=404, detail="Run not found")
        summary = stored.summary_json or build_run_summary(stored.run_type, stored.result_json)
        highlights = _extract_top_losses(stored.result_json, top_n=5)
        graph_evidence = build_graph_evidence(stored.result_json)
        evidence = {
            "run_id": stored.run_id,
            "run_type": stored.run_type,
            "created_at": stored.created_at,
            "request": stored.request_json,
            "summary": summary,
            "highlights": highlights,
            "graphs": graph_evidence,
        }
        prompt = build_prompt(req.question or "Summarize the simulation.", evidence)
        response = call_groq(prompt)
        return {"run_id": stored.run_id, "response": response, "evidence": evidence}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/explain/bank")
async def explain_bank(req: ExplainBankRequest):
    try:
        bank_profile = _llm_store.get_bank_profile(req.bank_id)
        if not bank_profile and req.bank_id:
            bank_profile = _llm_store.find_bank_profile_by_name(req.bank_id)
        if not bank_profile:
            raise HTTPException(status_code=404, detail="Bank profile not found")
        all_banks = _llm_store.get_all_bank_profiles()
        risk_scores = [float(b.get("gnn_risk_score", 0) or 0) for b in all_banks]
        assets = [float(b.get("total_assets", 0) or 0) for b in all_banks]
        leverage = [float(b.get("leverage_ratio", 0) or 0) for b in all_banks]
        bank_percentiles = {
            "risk_score_pct": _percentile_rank(risk_scores, float(bank_profile.get("gnn_risk_score", 0) or 0)),
            "assets_pct": _percentile_rank(assets, float(bank_profile.get("total_assets", 0) or 0)),
            "leverage_pct": _percentile_rank(leverage, float(bank_profile.get("leverage_ratio", 0) or 0)),
        }
        peer_summary = _bank_peer_summary(all_banks, bank_profile)
        stored = _llm_store.get_run(req.run_id) if req.run_id else None
        run_type = stored.run_type if stored else None
        run_results = stored.result_json if stored else None
        bank_context = build_bank_context(bank_profile, run_type, run_results)
        evidence = {
            "bank_id": req.bank_id,
            "bank_context": bank_context,
            "run_id": stored.run_id if stored else None,
            "bank_percentiles": bank_percentiles,
            "peer_summary": peer_summary,
        }
        prompt = build_bank_prompt(req.question or "Explain this bank.", evidence)
        response = call_groq(prompt)
        return {"bank_id": req.bank_id, "response": response, "evidence": evidence}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages is required")
        evidence = _build_chat_evidence(req)
        response_text = call_groq_chat(req.messages, evidence)
        return {"response": response_text, "evidence": evidence}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
@app.get("/api/debug/ml")
async def debug_ml():
    """Debug endpoint to check ML availability and model status."""
    model = _get_cached_model()
    norm_path = Path(__file__).parent / "norm_stats.pt"
    return {
        "ML_AVAILABLE": ML_AVAILABLE,
        "ML_ERROR": ML_ERROR,
        "GNN_MODEL_PATH": str(GNN_MODEL_PATH),
        "GNN_MODEL_EXISTS": GNN_MODEL_PATH.exists(),
        "NORM_STATS_PATH": str(norm_path),
        "NORM_STATS_EXISTS": norm_path.exists(),
        "MODEL_LOADED": model is not None,
        "MODEL_TYPE": str(type(model)) if model else None,
        "CACHE_KEYS": list(_cache.keys()),
        "GNN_MODEL_ERROR_CACHE": _cache.get("gnn_model_error")
    }