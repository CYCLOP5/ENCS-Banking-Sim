"""
api.py — FastAPI REST wrapper for the ENCS Systemic Risk Engine
================================================================
Exposes the simulation, game-theory, and climate modules over HTTP
so the React frontend can consume them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import traceback

import simulation_engine as sim
from strategic_model import run_game_simulation
from climate_risk import assign_climate_exposure, run_transition_shock

try:
    import torch
    from ml_pipeline import load_trained_model, predict_risk
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from pathlib import Path

GNN_MODEL_PATH = Path(__file__).parent / "gnn_model.pth"

# ── App ────────────────────────────────────────────────────────────────────

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

# ── Cached network data ───────────────────────────────────────────────────

_cache: Dict[str, Any] = {}


def _load_network():
    """Load network once and cache."""
    if "W_dense" not in _cache:
        W_sparse, df = sim.load_and_align_network()
        W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)
        _cache["W_sparse"] = W_sparse
        _cache["W_dense"] = W_dense
        _cache["df"] = df
    return _cache["W_dense"], _cache["df"]


# ── Request / Response schemas ────────────────────────────────────────────

class SimulationRequest(BaseModel):
    trigger_idx: int = 0
    severity: float = Field(1.0, ge=0, le=1)
    max_iter: int = Field(100, ge=10, le=500)
    tolerance: float = 1e-5
    distress_threshold: float = Field(0.95, ge=0, le=1)
    # intraday
    use_intraday: bool = True
    n_steps: int = Field(10, ge=1, le=50)
    sigma: float = Field(0.05, ge=0.01, le=0.30)
    panic_rate: float = Field(0.10, ge=0, le=0.50)
    fire_sale_alpha: float = Field(0.005, ge=0, le=0.05)
    margin_multiplier: float = Field(1.0, ge=0, le=5)
    # CCP
    use_ccp: bool = False
    clearing_rate: float = Field(0.5, ge=0, le=1)
    default_fund_ratio: float = Field(0.05, ge=0.01, le=0.25)


class ClimateRequest(BaseModel):
    carbon_tax: float = Field(0.5, ge=0, le=1)
    green_subsidy: float = Field(0.10, ge=0, le=0.50)
    use_intraday: bool = True
    trigger_idx: int = 0
    severity: float = Field(1.0, ge=0, le=1)
    n_steps: int = Field(10, ge=1, le=50)


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


# ── Helpers ───────────────────────────────────────────────────────────────

def _ndarray_to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _clean_results(d: dict) -> dict:
    """Convert numpy arrays and other non-serializable types to JSON-safe."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _clean_results(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            out[k] = [x.tolist() for x in v]
        elif k == "agents":
            # Skip non-serialisable agent objects
            continue
        else:
            out[k] = v
    return out


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "ml_available": ML_AVAILABLE}


@app.get("/api/topology")
async def get_topology():
    """
    Return the network graph as JSON suitable for react-force-graph-3d.
    Nodes = banks, Links = interbank edges.
    """
    try:
        W_dense, df = _load_network()
        n = W_dense.shape[0]

        # Build nodes
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

        # Build links (top edges by weight to avoid 50k+ links)
        rows_idx, cols_idx = np.nonzero(W_dense)
        weights = W_dense[rows_idx, cols_idx]
        # Keep top ~2000 edges
        if len(weights) > 2000:
            threshold = np.percentile(weights, 100 * (1 - 2000 / len(weights)))
            mask = weights >= threshold
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

        # Enrich with climate data
        df_enriched = df.copy()
        assign_climate_exposure(df_enriched)

        # GNN risk scores
        risk_scores = np.zeros(len(df_enriched))
        if ML_AVAILABLE and GNN_MODEL_PATH.exists():
            try:
                model = load_trained_model(str(GNN_MODEL_PATH))
                result = predict_risk(model, W_dense, df_enriched)
                risk_scores = result["risk_scores"]
            except Exception:
                pass

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

        return {"banks": banks, "total": len(banks)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate")
async def run_simulation(req: SimulationRequest):
    """Run the Eisenberg-Noe / Intraday simulation."""
    try:
        W_dense, df = _load_network()

        # Optional CCP
        if req.use_ccp:
            W_dense, df = sim.apply_central_clearing(
                W_dense, df,
                clearing_rate=req.clearing_rate,
                default_fund_ratio=req.default_fund_ratio,
            )

        state = sim.compute_state_variables(W_dense, df)

        if req.use_intraday:
            results = sim.run_rust_intraday(
                state, df,
                trigger_idx=req.trigger_idx,
                loss_severity=req.severity,
                n_steps=req.n_steps,
                sigma=req.sigma,
                panic_rate=req.panic_rate,
                fire_sale_alpha=req.fire_sale_alpha,
                margin_multiplier=req.margin_multiplier,
                max_iterations=req.max_iter,
                convergence_threshold=req.tolerance,
                distress_threshold=req.distress_threshold,
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
        return _clean_results(results)
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

        results = run_transition_shock(
            df=df,
            W_dense=W_dense,
            carbon_tax_severity=req.carbon_tax,
            green_subsidy=req.green_subsidy,
            use_intraday=req.use_intraday,
            trigger_idx=req.trigger_idx,
            loss_severity=req.severity,
            n_steps=req.n_steps,
        )

        results["bank_names"] = df["bank_name"].tolist()
        return _clean_results(results)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/game")
async def run_game(req: GameRequest):
    """Run Morris & Shin strategic game under both regimes."""
    try:
        results_opaque = run_game_simulation(
            info_regime="OPAQUE",
            n_banks=req.n_banks,
            n_steps=req.n_steps,
            true_solvency=req.true_solvency,
            interest_rate=req.interest_rate,
            recovery_rate=req.recovery_rate,
            risk_aversion_mean=req.risk_aversion,
            private_noise_std=req.noise_std,
            fire_sale_haircut=req.haircut,
            margin_pressure_rate=req.margin_pressure,
            interbank_exposure_usd=req.exposure,
        )

        results_transparent = run_game_simulation(
            info_regime="TRANSPARENT",
            n_banks=req.n_banks,
            n_steps=req.n_steps,
            true_solvency=req.true_solvency,
            interest_rate=req.interest_rate,
            recovery_rate=req.recovery_rate,
            risk_aversion_mean=req.risk_aversion,
            private_noise_std=req.noise_std,
            fire_sale_haircut=req.haircut,
            margin_pressure_rate=req.margin_pressure,
            interbank_exposure_usd=req.exposure,
        )

        return {
            "opaque": _clean_results(results_opaque),
            "transparent": _clean_results(results_transparent),
            "capital_saved": float(
                results_opaque.get("total_fire_sale_loss", 0)
                - results_transparent.get("total_fire_sale_loss", 0)
            ),
        }
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


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
