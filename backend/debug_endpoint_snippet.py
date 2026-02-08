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
