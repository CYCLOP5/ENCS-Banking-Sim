
import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add current directory to path so we can import modules
sys.path.append(str(Path(__file__).parent))

import simulation_engine as sim
import ml_pipeline as ml

def main():
    print("DEBUG: Starting GNN inference debug...")
    
    # Check paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "gnn_model.pth"
    norm_path = base_dir / "norm_stats.pt"
    
    print(f"DEBUG: Model path: {model_path} (Exists: {model_path.exists()})")
    print(f"DEBUG: Norm path: {norm_path} (Exists: {norm_path.exists()})")
    
    # Load data
    print("DEBUG: Loading network data...")
    W_sparse, df = sim.load_and_align_network()
    W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)
    
    # Load model
    print("DEBUG: Loading model...")
    try:
        model = ml.load_trained_model(str(model_path))
        print("DEBUG: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    # Run inference
    print("DEBUG: Running predict_risk...")
    try:
        # We need to reproduce what predict_risk does to see intermediates if needed,
        # but let's first call it and see the output.
        result = ml.predict_risk(model, W_dense, df)
        
        risk_probs = result["risk_probs"]
        risk_labels = result["risk_labels"]
        
        print(f"DEBUG: Output shape: {risk_probs.shape}")
        print(f"DEBUG: Min prob: {risk_probs.min()}")
        print(f"DEBUG: Max prob: {risk_probs.max()}")
        print(f"DEBUG: Mean prob: {risk_probs.mean()}")
        print(f"DEBUG: Standard deviation: {risk_probs.std()}")
        
        # Check for 0.0s
        zeros = (risk_probs == 0.0).sum()
        print(f"DEBUG: Number of 0.0s: {zeros} / {len(risk_probs)}")
        
        # Print top 10
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
