import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).parent))
import simulation_engine as sim
import ml_pipeline as ml
def main():
    print("DEBUG: Starting GNN inference debug...")
    base_dir = Path(__file__).parent
    model_path = base_dir / "gnn_model.pth"
    norm_path = base_dir / "norm_stats.pt"
    print(f"DEBUG: Model path: {model_path} (Exists: {model_path.exists()})")
    print(f"DEBUG: Norm path: {norm_path} (Exists: {norm_path.exists()})")
    print("DEBUG: Loading network data...")
    W_sparse, df = sim.load_and_align_network()
    W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)
    print("DEBUG: Loading model...")
    try:
        model = ml.load_trained_model(str(model_path))
        print("DEBUG: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    print("DEBUG: Running predict_risk...")
    try:
        result = ml.predict_risk(model, W_dense, df)
        risk_probs = result["risk_probs"]
        risk_labels = result["risk_labels"]
        print(f"DEBUG: Output shape: {risk_probs.shape}")
        print(f"DEBUG: Min prob: {risk_probs.min()}")
        print(f"DEBUG: Max prob: {risk_probs.max()}")
        print(f"DEBUG: Mean prob: {risk_probs.mean()}")
        print(f"DEBUG: Standard deviation: {risk_probs.std()}")
        zeros = (risk_probs == 0.0).sum()
        print(f"DEBUG: Number of 0.0s: {zeros} / {len(risk_probs)}")
        top10_idx = np.argsort(risk_probs)[::-1][:10]
        print("DEBUG: Top 10 risky banks:")
        for idx in top10_idx:
            print(f"  {df.iloc[idx]['bank_name']}: {risk_probs[idx]}")
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()