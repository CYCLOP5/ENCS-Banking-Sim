import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
BASE_PATH = Path(__file__).parent / "data"
OUTPUT_DIR = BASE_PATH / "output"
EU_INTERBANK_RATIO = 0.15  
US_INTERBANK_RATIO = 0.10  
DERIV_MULTIPLIER = 1.5
SUPER_CORE_COUNT = 30  
CORE_PERCENTILE = 0.95  
DISTANCE_SAME_REGION = 1.0
DISTANCE_CROSS_REGION = 10.0
RAS_ITERATIONS = 20
RAS_TOLERANCE = 1e-6
def load_and_impute_exposures() -> pd.DataFrame:
    """
    Load master nodes and impute interbank assets/liabilities.
    EU Banks: 15% of Total Assets/Liabilities
    US Banks: 40% (scaled up for missing holding companies)
    Risk Multiplier: 1.5x if derivatives exposure > 0
    """
    print("=" * 60)
    print("STEP 1: LOADING DATA & IMPUTING INTERBANK EXPOSURES")
    print("=" * 60)
    master_path = OUTPUT_DIR / "master_nodes.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Master nodes not found: {master_path}")
    df = pd.read_csv(master_path)
    print(f"  Loaded {len(df)} banks")
    df['interbank_ratio'] = np.where(df['region'] == 'EU', EU_INTERBANK_RATIO, US_INTERBANK_RATIO)
    df['interbank_assets'] = df['total_assets'] * df['interbank_ratio']
    df['interbank_liabilities'] = df['total_liabilities'] * df['interbank_ratio']
    if 'deriv_ir_notional' in df.columns:
        has_deriv = df['deriv_ir_notional'] > 0
        df.loc[has_deriv, 'interbank_assets'] *= DERIV_MULTIPLIER
        df.loc[has_deriv, 'interbank_liabilities'] *= DERIV_MULTIPLIER
        print(f"  Applied {DERIV_MULTIPLIER}x multiplier to {has_deriv.sum()} banks with derivatives")
    print(f"\n  Interbank Exposures Summary:")
    for region in ['US', 'EU']:
        mask = df['region'] == region
        total_ib_assets = df.loc[mask, 'interbank_assets'].sum() / 1e12
        total_ib_liabs = df.loc[mask, 'interbank_liabilities'].sum() / 1e12
        print(f"    {region}: Assets=${total_ib_assets:.2f}T, Liabilities=${total_ib_liabs:.2f}T")
    return df
def classify_core_periphery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify banks into tiers:
    - Super Core: Top 30 by interbank assets (global systemically important)
    - Core: Top 5% of remaining banks
    - Periphery: Everyone else
    """
    print("\n" + "=" * 60)
    print("STEP 2: CORE-PERIPHERY CLASSIFICATION")
    print("=" * 60)
    df = df.sort_values('interbank_assets', ascending=False).reset_index(drop=True)
    n = len(df)
    super_core_idx = min(SUPER_CORE_COUNT, n)
    remaining_n = n - super_core_idx
    core_count = int(remaining_n * (1 - CORE_PERCENTILE))
    core_idx = super_core_idx + core_count
    df['tier'] = 'periphery'
    df.loc[:super_core_idx-1, 'tier'] = 'super_core'
    df.loc[super_core_idx:core_idx-1, 'tier'] = 'core'
    df['tier_rank'] = np.arange(n)
    tier_counts = df['tier'].value_counts()
    print(f"  Super Core: {tier_counts.get('super_core', 0)} banks")
    print(f"  Core: {tier_counts.get('core', 0)} banks")
    print(f"  Periphery: {tier_counts.get('periphery', 0)} banks")
    print("\n  Top 10 Super Core Banks:")
    for i, row in df.head(10).iterrows():
        print(f"    {i+1}. {row['bank_name'][:40]:40s} ${row['interbank_assets']/1e12:.2f}T")
    return df
def build_gravity_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    """
    Construct adjacency matrix using Gravity Model.
    Weight w_ij = (Liab_i * Asset_j) / Distance
    i = debtor (borrower), j = creditor (lender)
    Connection rules:
    - Super Core connects to everyone
    - Core connects to Core + Super Core
    - Periphery connects only to Core + Super Core
    Distance:
    - Same region = 1.0
    - Cross region = 10.0
    """
    print("\n" + "=" * 60)
    print("STEP 3: GRAVITY MODEL MATRIX CONSTRUCTION")
    print("=" * 60)
    n = len(df)
    print(f"  Building {n}x{n} sparse matrix...")
    assets = df['interbank_assets'].values
    liabs = df['interbank_liabilities'].values
    regions = df['region'].values
    tiers = df['tier'].values
    rows = []
    cols = []
    weights = []
    super_core_mask = tiers == 'super_core'
    core_mask = tiers == 'core'
    periphery_mask = tiers == 'periphery'
    super_core_idx = np.where(super_core_mask)[0]
    core_idx = np.where(core_mask)[0]
    print(f"  Computing connections...")
    for i in range(n):
        if i % 500 == 0:
            print(f"    Processing bank {i}/{n}...")
        tier_i = tiers[i]
        region_i = regions[i]
        asset_i = assets[i]
        if asset_i <= 0:
            continue
        if tier_i == 'super_core':
            targets = np.arange(n)
        elif tier_i == 'core':
            targets = np.concatenate([super_core_idx, core_idx])
        else:
            targets = np.concatenate([super_core_idx, core_idx])
        targets = targets[targets != i]
        liab_i = liabs[i]
        if liab_i <= 0:
            continue
        for j in targets:
            asset_j = assets[j]
            if asset_j <= 0:
                continue
            if regions[i] == regions[j]:
                distance = DISTANCE_SAME_REGION
            else:
                distance = DISTANCE_CROSS_REGION
            weight = (liab_i * asset_j) / distance
            if weight > 0:
                rows.append(i)
                cols.append(j)
                weights.append(weight)
    print(f"  Created {len(weights)} edges")
    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n))
    W = W.tocsr()
    max_weight = W.max()
    if max_weight > 0:
        W = W / max_weight
    print(f"  Matrix density: {W.nnz / (n*n) * 100:.4f}%")
    return W
def ras_balance(W: sparse.csr_matrix, target_row_sums: np.ndarray, 
                target_col_sums: np.ndarray, iterations: int = RAS_ITERATIONS) -> sparse.csr_matrix:
    """
    Apply RAS/Sinkhorn-Knopp algorithm to balance the matrix.
    Iteratively adjusts W so that:
    - Row sums ≈ interbank_liabilities (outflows / obligations)
    - Col sums ≈ interbank_assets      (inflows / claims)
    """
    print("\n" + "=" * 60)
    print("STEP 4: RAS (SINKHORN-KNOPP) BALANCING")
    print("=" * 60)
    W_dense = W.toarray()
    n = W_dense.shape[0]
    target_row_sums = target_row_sums / target_row_sums.sum() if target_row_sums.sum() > 0 else target_row_sums
    target_col_sums = target_col_sums / target_col_sums.sum() if target_col_sums.sum() > 0 else target_col_sums
    print(f"  Running {iterations} RAS iterations...")
    for it in range(iterations):
        row_sums = W_dense.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1)  
        r_factors = target_row_sums / row_sums
        W_dense = W_dense * r_factors[:, np.newaxis]
        col_sums = W_dense.sum(axis=0)
        col_sums = np.where(col_sums > 0, col_sums, 1)
        s_factors = target_col_sums / col_sums
        W_dense = W_dense * s_factors[np.newaxis, :]
        if it % 5 == 0:
            row_err = np.abs(W_dense.sum(axis=1) - target_row_sums).mean()
            col_err = np.abs(W_dense.sum(axis=0) - target_col_sums).mean()
            print(f"    Iteration {it}: Row error={row_err:.6f}, Col error={col_err:.6f}")
    final_row_err = np.abs(W_dense.sum(axis=1) - target_row_sums).mean()
    final_col_err = np.abs(W_dense.sum(axis=0) - target_col_sums).mean()
    print(f"  Final: Row error={final_row_err:.6f}, Col error={final_col_err:.6f}")
    return sparse.csr_matrix(W_dense)
def export_outputs(W: sparse.csr_matrix, df: pd.DataFrame):
    """
    Export adjacency matrix and network map for visualization.
    """
    print("\n" + "=" * 60)
    print("STEP 5: EXPORTING OUTPUTS")
    print("=" * 60)
    OUTPUT_DIR.mkdir(exist_ok=True)
    matrix_path = OUTPUT_DIR / "adjacency_matrix.npz"
    sparse.save_npz(matrix_path, W)
    print(f"  Saved: {matrix_path}")
    print(f"    → Shape: {W.shape}, Non-zeros: {W.nnz}")
    nodes = []
    for i, row in df.iterrows():
        x = 0 if row['region'] == 'US' else 100
        x += np.random.uniform(-20, 20)
        if row['tier'] == 'super_core':
            y = 100
        elif row['tier'] == 'core':
            y = 50
        else:
            y = 10
        y += np.random.uniform(-5, 5)
        y *= np.log10(row['interbank_assets'] + 1) / 15  
        nodes.append({
            'id': str(row['bank_id']),
            'name': str(row['bank_name'])[:50] if pd.notna(row['bank_name']) else str(row['bank_id'])[:50],
            'region': row['region'],
            'tier': row['tier'],
            'interbank_assets': float(row['interbank_assets']),
            'x': float(x),
            'y': float(y)
        })
    W_coo = W.tocoo()
    edge_data = list(zip(W_coo.row, W_coo.col, W_coo.data))
    edge_data.sort(key=lambda x: -x[2])  
    edges = []
    for src, tgt, weight in edge_data[:5000]:
        edges.append({
            'source': str(df.iloc[src]['bank_id']),
            'target': str(df.iloc[tgt]['bank_id']),
            'weight': float(weight)
        })
    network_map = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_nodes': len(nodes),
            'total_edges': W.nnz,
            'edges_exported': len(edges)
        }
    }
    map_path = OUTPUT_DIR / "network_map.json"
    with open(map_path, 'w') as f:
        json.dump(network_map, f, indent=2)
    print(f"  Saved: {map_path}")
    print(f"    → {len(nodes)} nodes, {len(edges)} edges (top 5000)")
    return network_map
def analyze_connectivity(W: sparse.csr_matrix, df: pd.DataFrame):
    """
    Analyze and compare US vs EU systemic connectivity.
    """
    print("\n" + "=" * 60)
    print("STEP 6: CONNECTIVITY ANALYSIS")
    print("=" * 60)
    us_mask = df['region'] == 'US'
    eu_mask = df['region'] == 'EU'
    us_idx = np.where(us_mask)[0]
    eu_idx = np.where(eu_mask)[0]
    W_dense = W.toarray()
    us_us = W_dense[np.ix_(us_idx, us_idx)].sum()
    eu_eu = W_dense[np.ix_(eu_idx, eu_idx)].sum()
    us_eu = W_dense[np.ix_(us_idx, eu_idx)].sum()
    eu_us = W_dense[np.ix_(eu_idx, us_idx)].sum()
    total = W_dense.sum()
    print(f"\n  === SYSTEMIC CONNECTIVITY BREAKDOWN ===")
    print(f"  US → US: {us_us/total*100:.2f}% of total weight")
    print(f"  EU → EU: {eu_eu/total*100:.2f}% of total weight")
    print(f"  US → EU: {us_eu/total*100:.2f}% of total weight")
    print(f"  EU → US: {eu_us/total*100:.2f}% of total weight")
    us_avg_out = W_dense[us_idx, :].sum(axis=1).mean()
    eu_avg_out = W_dense[eu_idx, :].sum(axis=1).mean()
    us_avg_in = W_dense[:, us_idx].sum(axis=0).mean()
    eu_avg_in = W_dense[:, eu_idx].sum(axis=0).mean()
    print(f"\n  === PER-BANK AVERAGE CONNECTIVITY ===")
    print(f"  US Bank Avg Outflow: {us_avg_out:.6f}")
    print(f"  EU Bank Avg Outflow: {eu_avg_out:.6f}")
    print(f"  US Bank Avg Inflow:  {us_avg_in:.6f}")
    print(f"  EU Bank Avg Inflow:  {eu_avg_in:.6f}")
    print(f"\n  === CONNECTIVITY BY TIER ===")
    for tier in ['super_core', 'core', 'periphery']:
        tier_idx = np.where(df['tier'] == tier)[0]
        avg_degree = (W_dense[tier_idx, :] > 0).sum(axis=1).mean()
        avg_weight = W_dense[tier_idx, :].sum(axis=1).mean()
        print(f"  {tier:12s}: Avg Degree={avg_degree:.1f}, Avg Weight={avg_weight:.6f}")
    print(f"\n  === NORMALIZATION VALIDATION ===")
    us_systemic_weight = (us_us + us_eu) / total * 100
    eu_systemic_weight = (eu_eu + eu_us) / total * 100
    print(f"  US Systemic Weight: {us_systemic_weight:.2f}%")
    print(f"  EU Systemic Weight: {eu_systemic_weight:.2f}%")
def main():
    print("\n" + "=" * 60)
    print("ENCS LAYER 2: TOPOLOGY RECONSTRUCTION")
    print("Interbank Network via Gravity Model + RAS")
    print("=" * 60)
    df = load_and_impute_exposures()
    df = classify_core_periphery(df)
    W = build_gravity_matrix(df)
    target_row = df['interbank_liabilities'].values
    target_col = df['interbank_assets'].values
    W_balanced = ras_balance(W, target_row, target_col)
    network_map = export_outputs(W_balanced, df)
    analyze_connectivity(W_balanced, df)
    print("\n" + "=" * 60)
    print("=" * 60)
    return W_balanced, df, network_map
if __name__ == "__main__":
    W, df, network_map = main()