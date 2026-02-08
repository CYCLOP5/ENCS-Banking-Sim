import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import norm
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

from strategic_model import EdgeStrategicAgent

try:
    import encs_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

BASE_PATH = Path(__file__).parent / "data"
OUTPUT_DIR = BASE_PATH / "output"
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_THRESHOLD = 1e-5
DEFAULT_DISTRESS_THRESHOLD = 0.5
EU_INTERBANK_RATIO = 0.15
US_INTERBANK_RATIO = 0.10  # Fixed: was 0.40, real-world is <10% for large banks
DERIV_MULTIPLIER = 1.5
DERIVATIVES_NETTING_RATIO = 0.005  # Proxies conversion from Gross Notional to Net Fair Value
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
def apply_central_clearing(W_dense: np.ndarray, df: pd.DataFrame,
                           clearing_rate: float = 0.5,
                           default_fund_ratio: float = 0.05) -> tuple:
    """
    LAYER 5 — CENTRAL CLEARING INFRASTRUCTURE
    Transform bilateral OTC topology into hub-and-spoke CCP topology.

    For each edge A->B with weight w:
      - With probability `clearing_rate`: remove A->B, add A->CCP and CCP->B
      - Otherwise: keep A->B as-is (bilateral residual)

    The CCP node receives a Default Fund = total_risk × default_fund_ratio.

    Returns:
        (W_new, df_new) — expanded matrix (N+1 × N+1) and dataframe with CCP row.
    """
    print("\n" + "=" * 60)
    print("APPLYING CENTRAL CLEARING (CCP TOPOLOGY)")
    print("=" * 60)

    n = W_dense.shape[0]
    ccp_idx = n  

    W_new = np.zeros((n + 1, n + 1), dtype=np.float64)

    rng = np.random.RandomState(seed=123)  

    cleared_volume = 0.0
    bilateral_volume = 0.0
    edges_cleared = 0
    edges_bilateral = 0

    for i in range(n):
        for j in range(n):
            w = W_dense[i, j]
            if w <= 0 or i == j:
                continue
            if rng.random() < clearing_rate:

                W_new[i, ccp_idx] += w      

                W_new[ccp_idx, j] += w       

                cleared_volume += w
                edges_cleared += 1
            else:
                W_new[i, j] = w
                bilateral_volume += w
                edges_bilateral += 1

    ccp_liabilities = W_new[ccp_idx, :].sum()  

    ccp_assets = W_new[:, ccp_idx].sum()        

    ccp_total_risk = max(ccp_liabilities, ccp_assets)
    ccp_equity = ccp_total_risk * default_fund_ratio
    ccp_total_assets = ccp_assets + ccp_equity  

    ccp_total_liabilities = ccp_liabilities

    ccp_row = {col: 0.0 for col in df.columns}
    ccp_row.update({
        'bank_id': 'CCP_GLOBAL',
        'bank_name': '⚡ Global CCP',
        'region': 'Global',
        'NSA': 'CCP',
        'total_assets': ccp_total_assets,
        'total_liabilities': ccp_total_liabilities,
        'equity_capital': ccp_equity,
        'leverage_ratio': 0.0,
    })
    df_new = pd.concat([df, pd.DataFrame([ccp_row])], ignore_index=True)

    df_new.loc[ccp_idx, 'interbank_assets'] = ccp_assets
    df_new.loc[ccp_idx, 'interbank_liabilities'] = ccp_liabilities

    # ── Deduct Default Fund contribution from member banks ──────────────
    # Each of the n original banks pre-funds an equal share.  Without this
    # the CCP's equity_capital appeared from thin air, violating
    # conservation of money.
    cost_per_bank = ccp_equity / max(n, 1)
    for i in range(n):
        df_new.loc[i, 'total_assets']    -= cost_per_bank
        df_new.loc[i, 'equity_capital']  -= cost_per_bank

    total_volume = cleared_volume + bilateral_volume
    print(f"  Original banks: {n}")
    print(f"  CCP node added at index: {ccp_idx}")
    print(f"  Edges cleared:  {edges_cleared} (${cleared_volume/1e12:.2f}T)")
    print(f"  Edges bilateral: {edges_bilateral} (${bilateral_volume/1e12:.2f}T)")
    print(f"  Clearing rate:  {cleared_volume/max(total_volume,1)*100:.1f}% of volume")
    print(f"  CCP Default Fund: ${ccp_equity/1e9:.1f}B ({default_fund_ratio*100:.1f}% of risk)")
    print(f"  Fund cost/bank:   ${cost_per_bank/1e9:.2f}B")
    print(f"  New matrix: {W_new.shape[0]}x{W_new.shape[1]}")

    return W_new, df_new

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

    if 'deriv_ir_notional' in df.columns:
        # Load raw Gross Notional, but scale down to Net Fair Value
        # to prevent unrealistic "Butterfly Effect" margin spirals.
        raw_notional = df['deriv_ir_notional'].fillna(0).values
        derivatives_exposure = raw_notional * DERIVATIVES_NETTING_RATIO
    else:
        derivatives_exposure = np.zeros(n)
    print(f"  Derivatives Exposure (Net): ${derivatives_exposure.sum() / 1e12:.2f}T ({(derivatives_exposure > 0).sum()} banks)")

    return {
        'obligations': obligations,
        'external_assets': external_assets.copy(),
        'external_liabilities': external_liabilities,
        'interbank_claims': interbank_claims,
        'equity': equity.copy(),
        'payments': obligations.copy(),
        'W': W_dense,
        'derivatives_exposure': derivatives_exposure,
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
def run_rust_intraday(state: dict, df: pd.DataFrame, trigger_idx: int,
                      loss_severity: float = 1.0,
                      n_steps: int = 10,
                      uncertainty_sigma: float = 0.05,
                      panic_threshold: float = 0.10,
                      alpha: float = 0.005,
                      max_iterations: int = DEFAULT_MAX_ITERATIONS,
                      convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
                      distress_threshold: float = DEFAULT_DISTRESS_THRESHOLD,
                      margin_sensitivity: float = 0.2,
                      circuit_breaker_threshold: float = 0.0) -> dict:
    """
    LAYER 4 — RUST INTRADAY ENGINE: Exponential fire sales + discrete time steps.

    If the Rust extension (encs_rust) is available, delegates the heavy computation
    to compiled Rust code.  Otherwise falls back to an equivalent pure-Python loop.

    Parameters:
        n_steps:            Number of discrete intraday time steps.
        uncertainty_sigma:  Gaussian noise on solvency signals.
        panic_threshold:    Signal level below which creditors run.
        alpha:              Exponential fire-sale decay constant.
    """
    print("\n" + "=" * 60)
    print("LAYER 4: INTRADAY SIMULATION" + (" (RUST)" if RUST_AVAILABLE else " (Python fallback)"))
    print("=" * 60)

    W = state['W'].copy()
    external_assets = state['external_assets'].copy()
    total_liabilities = df['total_liabilities'].values.copy()
    total_assets = df['total_assets'].values.copy()
    derivatives_exposure = state.get('derivatives_exposure', np.zeros(len(external_assets))).copy()
    n = len(external_assets)

    trigger_name = (str(df.iloc[trigger_idx]['bank_name'])[:40]
                    if pd.notna(df.iloc[trigger_idx]['bank_name']) else 'Unknown')

    print(f"  Trigger:       {trigger_name}")
    print(f"  Severity:      {loss_severity*100:.0f}%")
    print(f"  Time Steps:    {n_steps}")
    print(f"  σ (noise):     {uncertainty_sigma}")
    print(f"  Panic thresh:  {panic_threshold}")
    print(f"  α (fire-sale): {alpha}")
    print(f"  Margin sens:   {margin_sensitivity}")
    print(f"  Deriv exposure: ${derivatives_exposure.sum() / 1e12:.2f}T")
    if circuit_breaker_threshold > 0:
        print(f"  Circuit breaker: {circuit_breaker_threshold:.0%} drop")

    if RUST_AVAILABLE:
        print("  Delegating to Rust core...")
        result_dict = encs_rust.run_full_simulation(
            W.astype(np.float64),
            external_assets.astype(np.float64),
            total_liabilities.astype(np.float64),
            total_assets.astype(np.float64),
            derivatives_exposure.astype(np.float64),
            trigger_idx,
            loss_severity,
            n_steps=n_steps,
            sigma=uncertainty_sigma,
            panic_threshold=panic_threshold,
            alpha=alpha,
            max_clearing_iter=max_iterations,
            convergence_tol=convergence_threshold,
            distress_threshold=distress_threshold,
            margin_sensitivity=margin_sensitivity,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )

        results = {
            'trigger_idx': trigger_idx,
            'trigger_name': trigger_name,
            'loss_severity': loss_severity,
            'rust_engine': True,
            'n_steps': result_dict['n_steps'],
            'final_asset_price': result_dict['final_asset_price'],
            'n_defaults': result_dict['n_defaults'],
            'n_distressed': result_dict['n_distressed'],
            'equity_loss': result_dict['equity_loss'],
            'status': np.array(result_dict['status'], dtype='<U10'),
            'final_equity': np.array(result_dict['final_equity']),
            'initial_equity': np.array(result_dict['initial_equity']),
            'payments': np.array(result_dict['payments']),

            'price_timeline': result_dict['price_timeline'],
            'defaults_timeline': result_dict['defaults_timeline'],
            'distressed_timeline': result_dict['distressed_timeline'],
            'withdrawn_timeline': result_dict['withdrawn_timeline'],
            'gridlock_timeline': result_dict['gridlock_timeline'],
            'equity_loss_timeline': result_dict['equity_loss_timeline'],
            'margin_calls_timeline': result_dict['margin_calls_timeline'],
            'circuit_breaker_triggered': result_dict.get('circuit_breaker_triggered', False),
            'circuit_breaker_step': result_dict.get('circuit_breaker_step', None),
        }

    else:
        print("  Running pure-Python fallback (build Rust for 10-50× speedup)...")

        shock = external_assets[trigger_idx] * loss_severity
        external_assets[trigger_idx] -= shock
        total_assets[trigger_idx] -= shock

        asset_price = 1.0
        price_timeline = []
        defaults_timeline = []
        distressed_timeline = []
        withdrawn_timeline = []
        gridlock_timeline = []
        equity_loss_timeline = []
        margin_calls_timeline = []
        systemic_credit_losses = 0.0   # margin defaults (not fire-sale pressure)
        cb_triggered = False
        cb_step = None
        cb_floor = 1.0 - circuit_breaker_threshold if circuit_breaker_threshold > 0 else 0.0

        for t in range(1, n_steps + 1):

            # ── Circuit Breaker: if price has hit the floor, halt all trading ──
            if cb_floor > 0 and asset_price <= cb_floor:
                if not cb_triggered:
                    cb_triggered = True
                    cb_step = t
                    print(f"  CIRCUIT BREAKER TRIGGERED at step {t}  "
                          f"(price {asset_price:.4f} <= floor {cb_floor:.4f})")
                # Price is frozen, no new withdrawals or fire sales
                price_timeline.append(float(asset_price))
                equity = total_assets - total_liabilities
                n_def = int(np.sum(equity < 0))
                equity_ratio = np.where(
                    state['equity'] > 0,
                    equity / np.maximum(state['equity'], 1e-12),
                    1.0
                )
                n_dis = int(np.sum((equity_ratio < distress_threshold) & (equity >= 0)))
                eq_loss = float(np.sum(np.abs(equity[equity < 0])))
                defaults_timeline.append(n_def)
                distressed_timeline.append(n_dis)
                withdrawn_timeline.append(0.0)
                gridlock_timeline.append(0)
                equity_loss_timeline.append(eq_loss)
                margin_calls_timeline.append(0.0)
                continue

            solvency = np.where(total_assets > 0,
                                (total_assets - total_liabilities) / total_assets, 0.0)
            np.random.seed(42 + t)
            noise = np.random.normal(0.0, uncertainty_sigma, size=(n, n))
            signals = solvency[np.newaxis, :] + noise

            run_matrix = signals < panic_threshold
            total_withdrawn_per_bank = np.zeros(n)
            total_received_per_bank = np.zeros(n)   

            total_withdrawn_global = 0.0

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if run_matrix[j, i] and W[i, j] > 0:
                        withdrawn = W[i, j]
                        total_withdrawn_per_bank[i] += withdrawn   

                        total_received_per_bank[j] += withdrawn    

                        total_withdrawn_global += withdrawn
                        W[i, j] = 0.0

            margin_calls_total = 0.0
            if margin_sensitivity > 0.0:
                price_drop = 1.0 - asset_price
                if price_drop > 0.0:
                    for i in range(n):
                        margin_call = derivatives_exposure[i] * price_drop * margin_sensitivity
                        if margin_call <= 0.0:
                            continue
                        margin_calls_total += margin_call
                        if external_assets[i] >= margin_call:
                            external_assets[i] -= margin_call
                        else:
                            shortfall = margin_call - external_assets[i]
                            external_assets[i] = 0.0
                            # FIX: Margin default is a credit loss, not a
                            # liquidity withdrawal.  Adding it to
                            # total_withdrawn_global would inflate fire-sale
                            # pressure and violate conservation of money.
                            systemic_credit_losses += shortfall

            total_volume_norm = total_withdrawn_global / 1e12
            asset_price *= np.exp(-alpha * total_volume_norm)

            for i in range(n):

                external_assets[i] += total_received_per_bank[i]

                if total_withdrawn_per_bank[i] > 0:
                    fire_cost = total_withdrawn_per_bank[i] / asset_price
                    external_assets[i] = max(external_assets[i] - fire_cost, 0.0)

                total_liabilities[i] -= total_withdrawn_per_bank[i]

                total_assets[i] = external_assets[i] * asset_price + W[:, i].sum()

            equity = total_assets - total_liabilities

            obligations = W.sum(axis=1)
            pi = np.zeros_like(W)
            for i in range(n):
                if obligations[i] > 0:
                    pi[i, :] = W[i, :] / obligations[i]

            payments = obligations.copy()
            for _ in range(max_iterations):
                old_p = payments.copy()
                inflows = pi.T @ payments
                wealth = external_assets + inflows
                payments = np.minimum(obligations, np.maximum(0, wealth))
                if np.abs(payments - old_p).sum() < convergence_threshold:
                    break

            failed = int(np.sum((obligations > 1e-6) & ((payments / np.maximum(obligations, 1e-12)) < 0.999)))

            n_def = int(np.sum(equity < 0))
            # Compare current equity to pre-shock initial equity for distress
            equity_ratio = np.where(
                state['equity'] > 0,
                equity / np.maximum(state['equity'], 1e-12),
                1.0
            )
            n_dis = int(np.sum((equity_ratio < distress_threshold) & (equity >= 0)))
            eq_loss = float(np.sum(np.abs(equity[equity < 0])))

            price_timeline.append(float(asset_price))
            defaults_timeline.append(n_def)
            distressed_timeline.append(n_dis)
            withdrawn_timeline.append(float(total_withdrawn_global))
            gridlock_timeline.append(failed)
            equity_loss_timeline.append(eq_loss)
            margin_calls_timeline.append(float(margin_calls_total))

        initial_equity = state['equity'].copy()
        final_equity = total_assets - total_liabilities
        equity_ratio_final = np.where(initial_equity > 0, final_equity / initial_equity, 1.0)
        status = np.array(['Safe'] * n, dtype='<U10')
        status[final_equity < 0] = 'Default'
        status[(equity_ratio_final < distress_threshold) & (final_equity >= 0)] = 'Distressed'

        n_defaults = int((status == 'Default').sum())
        n_distressed = int((status == 'Distressed').sum())
        total_lost = float(np.sum(np.maximum(initial_equity - final_equity, 0)))

        results = {
            'trigger_idx': trigger_idx,
            'trigger_name': trigger_name,
            'loss_severity': loss_severity,
            'rust_engine': False,
            'n_steps': n_steps,
            'final_asset_price': asset_price,
            'n_defaults': n_defaults,
            'n_distressed': n_distressed,
            'equity_loss': total_lost,
            'status': status,
            'final_equity': final_equity,
            'initial_equity': initial_equity,
            'payments': payments,
            'price_timeline': price_timeline,
            'defaults_timeline': defaults_timeline,
            'distressed_timeline': distressed_timeline,
            'withdrawn_timeline': withdrawn_timeline,
            'gridlock_timeline': gridlock_timeline,
            'equity_loss_timeline': equity_loss_timeline,
            'margin_calls_timeline': margin_calls_timeline,
            'systemic_credit_losses': systemic_credit_losses,
            'circuit_breaker_triggered': cb_triggered,
            'circuit_breaker_step': cb_step,
        }

    print(f"\n  === RESULTS ===")
    print(f"  Engine:       {'Rust' if results.get('rust_engine') else 'Python'}")
    print(f"  Time Steps:   {results['n_steps']}")
    print(f"  Final Price:  {results['final_asset_price']:.4f}")
    print(f"  Defaults:     {results['n_defaults']}")
    print(f"  Distressed:   {results['n_distressed']}")
    print(f"  Capital Lost: ${results['equity_loss'] / 1e12:.2f}T")
    if results.get('systemic_credit_losses', 0) > 0:
        print(f"  Margin Defaults: ${results['systemic_credit_losses'] / 1e9:.1f}B (credit loss, not fire-sale)")
    if results.get('circuit_breaker_triggered'):
        print(f"  Circuit breaker halted trading at step {results['circuit_breaker_step']}")

    return results

# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGIC INTRADAY ENGINE  — replaces static panic_threshold with
#  per-edge Bayesian agents (Morris & Shin 1998 global-games framework).
# ═══════════════════════════════════════════════════════════════════════════

def run_strategic_intraday_simulation(
    state: dict,
    df: pd.DataFrame,
    trigger_idx: int,
    loss_severity: float = 1.0,
    n_steps: int = 10,
    uncertainty_sigma: float = 0.05,
    alpha: float = 0.005,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    distress_threshold: float = DEFAULT_DISTRESS_THRESHOLD,
    margin_sensitivity: float = 0.2,
    circuit_breaker_threshold: float = 0.0,
    # Strategic-model knobs  (sensible defaults match strategic_model.py)
    interest_rate: float = 0.05,
    recovery_rate: float = 0.40,
    risk_aversion_mean: float = 1.0,
    risk_aversion_std: float = 0.3,
    info_regime: str = "OPAQUE",
    public_precision: float = None,  # Explicit α override
    seed: int = 42,
) -> dict:
    """
    LAYER 4b — STRATEGIC INTRADAY SIMULATION
    =========================================
    Drop-in replacement for the Python fallback in ``run_rust_intraday``.

    **Key difference:** instead of ``signals < panic_threshold`` (a static,
    global scalar), every directed edge A→B is governed by an
    :class:`EdgeStrategicAgent` that:

      1. Observes a **noisy private signal** of Bank B's *specific* solvency.
      2. Observes a **public signal** (quality depends on ``info_regime``).
      3. Forms a Bayesian posterior  P(B defaults).
      4. Computes expected utility  U_stay  vs  U_withdraw.
      5. Decides per-edge whether to pull funding  W[A, B].

    This closes the logical gap: withdrawal decisions are now *endogenous*
    and heterogeneous — Bank A might trust B but panic about C.

    Everything else (fire sales, margin calls, Eisenberg-Noe clearing,
    circuit breaker, return schema) is identical to the existing engine
    so the API remains a drop-in.

    Parameters match ``run_rust_intraday`` plus the strategic knobs.
    """
    print("\n" + "=" * 60)
    print("LAYER 4b: STRATEGIC INTRADAY SIMULATION")
    print("=" * 60)

    rng = np.random.RandomState(seed)

    W = state['W'].copy()
    external_assets = state['external_assets'].copy()
    total_liabilities = df['total_liabilities'].values.copy()
    total_assets = df['total_assets'].values.copy()
    derivatives_exposure = state.get(
        'derivatives_exposure', np.zeros(len(external_assets))
    ).copy()
    n = len(external_assets)

    trigger_name = (
        str(df.iloc[trigger_idx]['bank_name'])[:40]
        if pd.notna(df.iloc[trigger_idx]['bank_name'])
        else 'Unknown'
    )


    # ── Signal precision (regime-dependent) ──────────────────────────────
    private_precision = 1.0 / (uncertainty_sigma ** 2)        # β
    
    if public_precision is not None:
        pass  # Use provided value
    elif info_regime == "TRANSPARENT":
        public_precision = 100.0                               # α
    else:  # OPAQUE
        public_precision = 0.01                                # α ≈ 0

    # ── Vectorized Setup ─────────────────────────────────────────────────
    # Identify all potentially active edges (non-zero weights)
    # We use arrays instead of a dict of objects for massive performance gain
    rows, cols = W.nonzero()
    # Filter out self-loops and zero weights (though nonzero() handles weights > 0 if W is positive)
    # Only strictly positive weights matter
    valid_mask = (rows != cols) & (W[rows, cols] > 0)
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    n_edges = len(rows)

    # Pre-generate risk aversion for each edge
    # This replaces the loop that created EdgeStrategicAgent objects
    edge_risk_aversions = np.maximum(0.1, rng.normal(risk_aversion_mean, risk_aversion_std, n_edges))

    print(f"  Trigger:           {trigger_name}")
    print(f"  Severity:          {loss_severity*100:.0f}%")
    print(f"  Time Steps:        {n_steps}")
    print(f"  σ (noise):         {uncertainty_sigma}")
    print(f"  α (fire-sale):     {alpha}")
    print(f"  Public Prec (α):   {public_precision}")
    print(f"  Margin sens:       {margin_sensitivity}")
    print(f"  Info regime:       {info_regime}")
    print(f"  Edge agents:       {n_edges} (Vectorized)")
    print(f"  Risk aversion μ:   {risk_aversion_mean}")
    if circuit_breaker_threshold > 0:
        print(f"  Circuit breaker:   {circuit_breaker_threshold:.0%} drop")

    # ── Apply initial shock ──────────────────────────────────────────────
    shock = external_assets[trigger_idx] * loss_severity
    external_assets[trigger_idx] -= shock
    total_assets[trigger_idx] -= shock

    asset_price = 1.0

    # ── Timeline accumulators (same schema as run_rust_intraday) ─────────
    price_timeline       = []
    defaults_timeline    = []
    distressed_timeline  = []
    withdrawn_timeline   = []
    gridlock_timeline    = []
    equity_loss_timeline = []
    margin_calls_timeline = []
    systemic_credit_losses = 0.0   # margin defaults (not fire-sale pressure)
    cb_triggered = False
    cb_step      = None
    cb_floor     = 1.0 - circuit_breaker_threshold if circuit_breaker_threshold > 0 else 0.0

    # ── Main intraday loop ───────────────────────────────────────────────
    for t in range(1, n_steps + 1):

        # ── Circuit breaker ──────────────────────────────────────────────
        if cb_floor > 0 and asset_price <= cb_floor:
            if not cb_triggered:
                cb_triggered = True
                cb_step = t
                print(f"  CIRCUIT BREAKER at step {t}  "
                      f"(price {asset_price:.4f} <= floor {cb_floor:.4f})")
            # Frozen step — record and continue
            price_timeline.append(float(asset_price))
            equity = total_assets - total_liabilities
            n_def = int(np.sum(equity < 0))
            eq_ratio = np.where(
                state['equity'] > 0,
                equity / np.maximum(state['equity'], 1e-12), 1.0
            )
            n_dis = int(np.sum((eq_ratio < distress_threshold) & (equity >= 0)))
            eq_loss = float(np.sum(np.abs(equity[equity < 0])))
            defaults_timeline.append(n_def)
            distressed_timeline.append(n_dis)
            withdrawn_timeline.append(0.0)
            gridlock_timeline.append(0)
            equity_loss_timeline.append(eq_loss)
            margin_calls_timeline.append(0.0)
            continue

        # ── Per-borrower solvency (public observable) ────────────────────
        solvency = np.where(
            total_assets > 0,
            (total_assets - total_liabilities) / total_assets,
            0.0,
        )

        # Public signal per borrower (regime-dependent quality)
        if info_regime == "TRANSPARENT":
            # Near-perfect public signal per node
            public_signals = solvency + rng.normal(0, 0.01, size=n)
        else:
            # Uninformative — falls back on private info
            public_signals = np.zeros(n)

        # ── Cumulative loss feedback (degrades effective solvency) ───────
        cum_loss_frac = float(np.sum(np.maximum(state['equity'] - (total_assets - total_liabilities), 0))
                              / max(np.sum(state['equity']), 1.0))
        margin_pressure = margin_sensitivity * 0.0 if asset_price >= 1.0 else \
            margin_sensitivity * (1.0 - asset_price) * 0.10

        # ── Vectorized Edge Decisions ─────────────────────────────────────
        total_withdrawn_per_bank = np.zeros(n)
        total_received_per_bank  = np.zeros(n)
        total_withdrawn_global   = 0.0

        # 1. Identify currently active edges from our pool
        #    (We use the original indices 'rows', 'cols', but check current W)
        #    Extract current exposures using fancy indexing
        current_exposures = W[rows, cols]
        
        # 2. Filter for edges that still have money (exposure > 0)
        active_indices = current_exposures > 1e-9
        
        if np.any(active_indices):
            # Subset arrays to just the active edges
            act_rows = rows[active_indices]
            act_cols = cols[active_indices]
            act_exposures = current_exposures[active_indices]
            act_risk_aversions = edge_risk_aversions[active_indices]
            n_active = len(act_rows)

            # 3. Vectorized Signal Generation
            #    Private signal: lender i's noisy observation of borrower j
            effective_solvency_j = solvency[act_cols] * (1.0 - 0.5 * cum_loss_frac)
            private_signals = effective_solvency_j + rng.normal(0, uncertainty_sigma, size=n_active)
            
            #    Public signals for these borrowers
            act_public_signals = public_signals[act_cols]

            # 4. Bayesian Posterior P(j defaults)
            posterior_mean = (
                public_precision * act_public_signals + 
                private_precision * private_signals
            ) / (public_precision + private_precision)
            
            posterior_std = 1.0 / np.sqrt(public_precision + private_precision)
            theta_star = 0.0
            p_defaults = norm.cdf((theta_star - posterior_mean) / posterior_std)

            # 5. Expected-utility decision
            #    Lender's current vs initial equity (for dynamic risk aversion)
            lender_cur_eq = total_assets[act_rows] - total_liabilities[act_rows]
            lender_init_eq = state['equity'][act_rows]
            
            #    Dynamic lambda: λ * (1 + 2 * loss_ratio)
            #    Avoid divide by zero
            equity_loss_ratios = np.zeros_like(lender_cur_eq)
            valid_init = lender_init_eq > 0
            equity_loss_ratios[valid_init] = 1.0 - (lender_cur_eq[valid_init] / lender_init_eq[valid_init])
            equity_loss_ratios = np.clip(equity_loss_ratios, 0.0, 1.0)
            
            effective_lambdas = act_risk_aversions * (1.0 + 2.0 * equity_loss_ratios)

            #    U_stay
            e_return_stay = (1.0 - p_defaults) * (1.0 + interest_rate) + p_defaults * recovery_rate
            spread = (1.0 + interest_rate) - recovery_rate
            variance_stay = p_defaults * (1.0 - p_defaults) * (spread ** 2)
            risk_stay = np.sqrt(variance_stay)
            U_stay = e_return_stay - effective_lambdas * risk_stay

            #    U_run
            U_run = 1.0 + margin_pressure

            # 6. Execute Withdrawals
            withdraw_mask = U_run > U_stay
            
            if np.any(withdraw_mask):
                w_rows = act_rows[withdraw_mask]
                w_cols = act_cols[withdraw_mask]
                w_amounts = act_exposures[withdraw_mask]

                # Update W (set to 0)
                W[w_rows, w_cols] = 0.0

                # Accumulate flows
                np.add.at(total_withdrawn_per_bank, w_cols, w_amounts)
                np.add.at(total_received_per_bank, w_rows, w_amounts)
                total_withdrawn_global = np.sum(w_amounts)

        # ── Margin calls on derivatives (unchanged from original) ────────
        margin_calls_total = 0.0
        if margin_sensitivity > 0.0:
            price_drop = 1.0 - asset_price
            if price_drop > 0.0:
                for i in range(n):
                    margin_call = derivatives_exposure[i] * price_drop * margin_sensitivity
                    if margin_call <= 0.0:
                        continue
                    margin_calls_total += margin_call
                    if external_assets[i] >= margin_call:
                        external_assets[i] -= margin_call
                    else:
                        shortfall = margin_call - external_assets[i]
                        external_assets[i] = 0.0
                        # FIX: Margin default is a credit loss, not a
                        # liquidity withdrawal — same fix as Python fallback.
                        systemic_credit_losses += shortfall

        # ── Fire-sale price impact ───────────────────────────────────────
        total_volume_norm = total_withdrawn_global / 1e12
        asset_price *= np.exp(-alpha * total_volume_norm)

        # ── Balance-sheet update (identical to original engine) ──────────
        for i in range(n):
            external_assets[i] += total_received_per_bank[i]
            if total_withdrawn_per_bank[i] > 0:
                fire_cost = total_withdrawn_per_bank[i] / asset_price
                external_assets[i] = max(external_assets[i] - fire_cost, 0.0)
            total_liabilities[i] -= total_withdrawn_per_bank[i]
            total_assets[i] = external_assets[i] * asset_price + W[:, i].sum()

        # ── Eisenberg-Noe clearing (unchanged) ───────────────────────────
        equity = total_assets - total_liabilities
        obligations = W.sum(axis=1)
        pi = np.zeros_like(W)
        for i in range(n):
            if obligations[i] > 0:
                pi[i, :] = W[i, :] / obligations[i]

        payments = obligations.copy()
        for _ in range(max_iterations):
            old_p = payments.copy()
            inflows = pi.T @ payments
            wealth = external_assets + inflows
            payments = np.minimum(obligations, np.maximum(0, wealth))
            if np.abs(payments - old_p).sum() < convergence_threshold:
                break

        failed = int(np.sum(
            (obligations > 1e-6)
            & ((payments / np.maximum(obligations, 1e-12)) < 0.999)
        ))

        n_def = int(np.sum(equity < 0))
        eq_ratio = np.where(
            state['equity'] > 0,
            equity / np.maximum(state['equity'], 1e-12), 1.0
        )
        n_dis = int(np.sum((eq_ratio < distress_threshold) & (equity >= 0)))
        eq_loss = float(np.sum(np.abs(equity[equity < 0])))

        # ── Record step ──────────────────────────────────────────────────
        price_timeline.append(float(asset_price))
        defaults_timeline.append(n_def)
        distressed_timeline.append(n_dis)
        withdrawn_timeline.append(float(total_withdrawn_global))
        gridlock_timeline.append(failed)
        equity_loss_timeline.append(eq_loss)
        margin_calls_timeline.append(float(margin_calls_total))

    # ── Final status classification ──────────────────────────────────────
    initial_equity = state['equity'].copy()
    final_equity   = total_assets - total_liabilities
    eq_ratio_final = np.where(
        initial_equity > 0, final_equity / initial_equity, 1.0
    )
    status = np.array(['Safe'] * n, dtype='<U10')
    status[final_equity < 0] = 'Default'
    status[(eq_ratio_final < distress_threshold) & (final_equity >= 0)] = 'Distressed'

    n_defaults   = int((status == 'Default').sum())
    n_distressed = int((status == 'Distressed').sum())
    total_lost   = float(np.sum(np.maximum(initial_equity - final_equity, 0)))

    results = {
        'trigger_idx':       trigger_idx,
        'trigger_name':      trigger_name,
        'loss_severity':     loss_severity,
        'rust_engine':       False,
        'strategic_engine':  True,
        'info_regime':       info_regime,
        'n_edge_agents':     len(edge_agents),
        'n_steps':           n_steps,
        'final_asset_price': asset_price,
        'n_defaults':        n_defaults,
        'n_distressed':      n_distressed,
        'equity_loss':       total_lost,
        'status':            status,
        'final_equity':      final_equity,
        'initial_equity':    initial_equity,
        'payments':          payments,
        'price_timeline':        price_timeline,
        'defaults_timeline':     defaults_timeline,
        'distressed_timeline':   distressed_timeline,
        'withdrawn_timeline':    withdrawn_timeline,
        'gridlock_timeline':     gridlock_timeline,
        'equity_loss_timeline':  equity_loss_timeline,
        'margin_calls_timeline': margin_calls_timeline,
        'systemic_credit_losses': systemic_credit_losses,
        'circuit_breaker_triggered': cb_triggered,
        'circuit_breaker_step':     cb_step,
    }

    print(f"\n  === RESULTS ===")
    print(f"  Engine:       Strategic Bayesian (Morris & Shin)")
    print(f"  Info regime:  {info_regime}")
    print(f"  Edge agents:  {len(edge_agents)}")
    print(f"  Time Steps:   {n_steps}")
    print(f"  Final Price:  {asset_price:.4f}")
    print(f"  Defaults:     {n_defaults}")
    print(f"  Distressed:   {n_distressed}")
    print(f"  Capital Lost: ${total_lost / 1e12:.2f}T")
    if systemic_credit_losses > 0:
        print(f"  Margin Defaults: ${systemic_credit_losses / 1e9:.1f}B (credit loss, not fire-sale)")
    if cb_triggered:
        print(f"   Circuit breaker halted trading at step {cb_step}")

    return results


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