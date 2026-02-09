import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.append(str(Path(__file__).parent))
import simulation_engine as sim
def check_data():
    print("Checking data integrity...")
    try:
        W_sparse, df = sim.load_and_align_network()
        print(f"Loaded W_sparse: shape={W_sparse.shape}, nnz={W_sparse.nnz}, sum={W_sparse.sum()}")
        print(f"Loaded df: shape={df.shape}")
        print("Interbank Liabilities stats:")
        print(df['interbank_liabilities'].describe())
        W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)
        print(f"Rescaled W_dense: shape={W_dense.shape}, sum={W_dense.sum()}")
        print(f"Max edge: {W_dense.max()}")
        print(f"Min non-zero edge: {W_dense[W_dense > 0].min() if W_dense.sum() > 0 else 0}")
        row_sums = W_dense.sum(axis=1)
        zeros = (row_sums == 0).sum()
        print(f"Rows with 0 sum: {zeros} / {len(row_sums)}")
        if W_dense.sum() == 0:
             print("CRITICAL: Matrix is all zeros!")
        else:
             print("Matrix looks populated.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    check_data()