import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')
BASE_PATH = Path(__file__).parent / "data"
OUTPUT_DIR = BASE_PATH / "output"
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_THRESHOLD = 1e-5
DEFAULT_DISTRESS_THRESHOLD = 0.5
EU_INTERBANK_RATIO = 0.15
US_INTERBANK_RATIO = 0.40
DERIV_MULTIPLIER = 1.5
def load_and_align_network():
    """
    Load adjacency matrix and node data with proper alignment.
    CRITICAL: Must sort nodes exactly as Layer 2 did when building the matrix.
    The matrix was built with banks sorted by interbank_assets descending.
    """
    print("=" * 60)
    print("LOADING AND ALIGNING NETWORK DATA")
    print("=" * 60)
    nodes_path = OUTPUT_DIR / "master_nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Node data not found: {nodes_path}")
    df = pd.read_csv(nodes_path)
    df['interbank_ratio'] = np.where(df['region'] == 'EU', EU_INTERBANK_RATIO, US_INTERBANK_RATIO)
    df['interbank_assets'] = df['total_assets'] * df['interbank_ratio']
    df['interbank_liabilities'] = df['total_liabilities'] * df['interbank_ratio']
    if 'deriv_ir_notional' in df.columns:
        has_deriv = df['deriv_ir_notional'] > 0
        df.loc[has_deriv, 'interbank_assets'] *= DERIV_MULTIPLIER
        df.loc[has_deriv, 'interbank_liabilities'] *= DERIV_MULTIPLIER
    df = df.sort_values('interbank_assets', ascending=False).reset_index(drop=True)
    print(f"  Nodes: {len(df)} banks (sorted by interbank_assets DESC)")
    matrix_path = OUTPUT_DIR / "adjacency_matrix.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Adjacency matrix not found: {matrix_path}")
    W = sparse.load_npz(matrix_path)
    print(f"  Matrix: {W.shape[0]}x{W.shape[1]}, {W.nnz} edges")
    return W, df
def rescale_matrix_to_dollars(W: sparse.csr_matrix, df: pd.DataFrame) -> np.ndarray:
    """
    FIX for Zero Dollar Bug: Rescale normalized matrix weights to actual dollars.
    The matrix was normalized (max=1) in Layer 2. We need to rescale so that:
    - Row sums = interbank_liabilities (what each bank owes)
    - Col sums = interbank_assets (what each bank is owed)
    We use the interbank_liabilities as the target.
    """
    print("\n" + "=" * 60)
    print("RESCALING MATRIX TO DOLLARS")
    print("=" * 60)
    W_dense = W.toarray()
    n = W_dense.shape[0]
    current_row_sums = W_dense.sum(axis=1)
    target_row_sums = df['interbank_liabilities'].values
    for i in range(n):
        if current_row_sums[i] > 0:
            scale = target_row_sums[i] / current_row_sums[i]
            W_dense[i, :] *= scale
    new_row_sums = W_dense.sum(axis=1)
    print(f"  Total Interbank Obligations: ${new_row_sums.sum() / 1e12:.2f}T")
    print(f"  Avg Obligation per bank: ${new_row_sums.mean() / 1e9:.2f}B")
    return W_dense
def compute_state_variables(W_dense: np.ndarray, df: pd.DataFrame):
    """
    Compute initial state variables for Eisenberg-Noe clearing.
    - Obligations (L): Row sums of W (what bank i owes)
    - External Assets (e): Total assets - interbank claims
    - Equity: Assets - Liabilities
    """
    print("\n" + "=" * 60)
    print("COMPUTING STATE VARIABLES")
    print("=" * 60)
    n = W_dense.shape[0]
    obligations = W_dense.sum(axis=1)
    interbank_claims = W_dense.sum(axis=0)
    total_assets = df['total_assets'].values
    external_assets = np.maximum(total_assets - interbank_claims, 0)
    total_liabilities = df['total_liabilities'].values
    external_liabilities = np.maximum(total_liabilities - obligations, 0)
    equity = total_assets - total_liabilities
    print(f"  Total External Assets: ${external_assets.sum() / 1e12:.2f}T")
    print(f"  Total Obligations: ${obligations.sum() / 1e12:.2f}T")
    print(f"  Total Initial Equity: ${equity.sum() / 1e12:.2f}T")
    if obligations.sum() < 1e6:
        print("  ⚠ WARNING: Obligations near zero - check matrix scaling!")
    return {
        'obligations': obligations,
        'external_assets': external_assets.copy(),
        'external_liabilities': external_liabilities,
        'interbank_claims': interbank_claims,
        'equity': equity.copy(),
        'payments': obligations.copy(),
        'W': W_dense
    }
def run_scenario(state: dict, df: pd.DataFrame, trigger_idx: int,
                 loss_severity: float = 1.0,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
                 distress_threshold: float = DEFAULT_DISTRESS_THRESHOLD) -> dict:
    """
    Run Eisenberg-Noe clearing after shocking a trigger bank.
    """
    print("\n" + "=" * 60)
    print("RUNNING EISENBERG-NOE CLEARING")
    print("=" * 60)
    obligations = state['obligations'].copy()
    external_assets = state['external_assets'].copy()
    W = state['W'].copy()
    n = len(obligations)
    original_external = external_assets[trigger_idx]
    shock_amount = original_external * loss_severity
    external_assets[trigger_idx] -= shock_amount

    trigger_obligation = obligations[trigger_idx]

    trigger_name = str(df.iloc[trigger_idx]['bank_name'])[:40] if pd.notna(df.iloc[trigger_idx]['bank_name']) else 'Unknown'
    print(f"  Trigger: {trigger_name}")
    print(f"  Severity: {loss_severity * 100:.0f}%")
    print(f"  Assets Destroyed: ${shock_amount / 1e9:.2f}B")
    print(f"  Obligations Defaulted: ${trigger_obligation * loss_severity / 1e9:.2f}B")

    payments = obligations.copy()
    payments[trigger_idx] = obligations[trigger_idx] * (1 - loss_severity)  

    pi = np.zeros_like(W)
    for i in range(n):
        if obligations[i] > 0:
            pi[i, :] = W[i, :] / obligations[i]
    print(f"\n  Clearing iterations (max={max_iterations})...")
    for iteration in range(max_iterations):
        old_payments = payments.copy()
        inflows = pi.T @ payments
        wealth = external_assets + inflows

        new_payments = np.minimum(obligations, np.maximum(0, wealth))
        new_payments[trigger_idx] = obligations[trigger_idx] * (1 - loss_severity)  

        payments = new_payments
        diff = np.abs(payments - old_payments).sum()
        if diff < convergence_threshold:
            print(f"    Converged at iteration {iteration + 1} (diff={diff:.2e})")
            break
    else:
        print(f"    Max iterations reached ({max_iterations})")
    final_inflows = pi.T @ payments
    final_wealth = external_assets + final_inflows

    expected_inflows = pi.T @ obligations  

    lost_inflows = expected_inflows - final_inflows  

    initial_equity = state['equity'].copy()
    final_equity = initial_equity - lost_inflows

    trigger_loss = initial_equity[trigger_idx] * loss_severity
    final_equity[trigger_idx] = initial_equity[trigger_idx] * (1 - loss_severity)

    equity_ratio = np.where(initial_equity > 0, final_equity / initial_equity, 1.0)

    # Use explicit dtype to avoid string truncation ('Safe'=4 chars, 'Distressed'=10 chars)
    status = np.array(['Safe'] * n, dtype='<U10')
    status[final_equity < 0] = 'Default'  

    status[(equity_ratio < distress_threshold) & (final_equity >= 0)] = 'Distressed'

    n_defaults = (status == 'Default').sum()
    n_distressed = (status == 'Distressed').sum()
    total_lost = lost_inflows.sum() + trigger_loss

    print(f"\n  === RESULTS ===")
    print(f"  Defaults: {n_defaults}")
    print(f"  Distressed: {n_distressed}")
    print(f"  Trigger Loss: ${trigger_loss / 1e12:.2f}T")
    print(f"  Contagion Loss: ${lost_inflows.sum() / 1e12:.2f}T")
    print(f"  Total Capital Lost: ${total_lost / 1e12:.2f}T")
    return {
        'trigger_idx': trigger_idx,
        'trigger_name': trigger_name,
        'loss_severity': loss_severity,
        'payments': payments,
        'final_equity': final_equity,
        'initial_equity': initial_equity,
        'status': status,
        'n_defaults': n_defaults,
        'n_distressed': n_distressed,
        'equity_loss': total_lost
    }
def find_most_dangerous(W_dense: np.ndarray, df: pd.DataFrame) -> int:
    """Find most dangerous node by out-strength (total obligations)."""
    print("\n" + "=" * 60)
    print("IDENTIFYING MOST DANGEROUS NODE")
    print("=" * 60)
    out_strength = W_dense.sum(axis=1)
    top_indices = np.argsort(out_strength)[::-1][:10]
    print("  Top 10 by Interbank Obligations:")
    for rank, idx in enumerate(top_indices):
        name = df.iloc[idx]['bank_name'][:40]
        obligation = out_strength[idx] / 1e9
        print(f"    {rank+1}. {name:40s} ${obligation:.1f}B")
    return top_indices[0]
def export_results(df: pd.DataFrame, results: dict):
    """Export simulation results."""
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    output = pd.DataFrame({
        'bank_id': df['bank_id'],
        'bank_name': df['bank_name'],
        'region': df['region'],
        'initial_equity': results['initial_equity'],
        'final_equity': results['final_equity'],
        'equity_loss': results['initial_equity'] - results['final_equity'],
        'status': results['status']
    })
    output = output.sort_values('equity_loss', ascending=False)
    output_path = OUTPUT_DIR / "simulation_results.csv"
    output.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print("\n  Top 10 Casualties:")
    for i, (_, row) in enumerate(output.head(10).iterrows()):
        loss = row['equity_loss'] / 1e9
        print(f"    {i+1}. {row['bank_name'][:35]:35s} ${loss:,.1f}B [{row['status']}]")
    return output
def main():
    parser = argparse.ArgumentParser(description='ENCS Systemic Risk Simulation')
    parser.add_argument('--target', type=str, help='Bank to shock (name or ID)')
    parser.add_argument('--severity', type=float, default=1.0, help='0-1 (default=1.0)')
    parser.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument('--tolerance', type=float, default=DEFAULT_CONVERGENCE_THRESHOLD)
    parser.add_argument('--distress', type=float, default=DEFAULT_DISTRESS_THRESHOLD)
    args = parser.parse_args()
    print("\n" + "=" * 60)
    print("=" * 60)
    W_sparse, df = load_and_align_network()
    W_dense = rescale_matrix_to_dollars(W_sparse, df)
    state = compute_state_variables(W_dense, df)
    if args.target:
        matches = df[df['bank_id'].str.contains(args.target, case=False, na=False)]
        if len(matches) == 0:
            matches = df[df['bank_name'].str.contains(args.target, case=False, na=False)]
        if len(matches) == 0:
            print(f"\n  ERROR: No bank matching '{args.target}'")
            return
        trigger_idx = matches.index[0]
        print(f"\n  Target: {df.iloc[trigger_idx]['bank_name'][:50]}")
    else:
        trigger_idx = find_most_dangerous(W_dense, df)
    results = run_scenario(
        state, df, trigger_idx, args.severity,
        max_iterations=args.max_iter,
        convergence_threshold=args.tolerance,
        distress_threshold=args.distress
    )
    output_df = export_results(df, results)
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\n  >>> SUMMARY <<<")
    print(f"  Trigger: {results['trigger_name']}")
    print(f"  Shock: {results['loss_severity']*100:.0f}%")
    print(f"  Cascade Defaults: {results['n_defaults']}")
    print(f"  Banks Distressed: {results['n_distressed']}")
    print(f"  Capital Vaporized: ${results['equity_loss']/1e12:.2f} TRILLION")
    defaults_by_region = output_df[output_df['status'] == 'Default']['region'].value_counts()
    if len(defaults_by_region) > 0:
        print(f"\n  Defaults by Region:")
        for region, count in defaults_by_region.items():
            print(f"    {region}: {count} banks")
    return results
if __name__ == "__main__":
    main()