"""
climate_risk.py — Green Swan Transition Risk Module
====================================================

Simulates a "Green Swan" event: a sudden global Carbon Tax or regulatory
shift that causes Brown Assets (fossil fuels) to become stranded, while
Green Assets (renewables) gain value.

Geography matters:
    US banks  → higher Brown Asset exposure  (mean carbon_score ≈ 0.60)
    EU banks  → lower  Brown Asset exposure  (mean carbon_score ≈ 0.30)

Tier-1 "Greenwashing" discount: the largest banks publicly pledge ESG
targets, slightly reducing their reported carbon score — but not enough
to avert the shock.

The net shock hits bank capital immediately and then propagates through
the interbank network via the existing Eisenberg-Noe / Intraday engine,
proving that climate risk is *systemic*: even "green" banks can fail
when their brown counterparties default.

Reference
---------
Bolton, P. et al. (2020). "The Green Swan: Central Banking and
Financial Stability in the Age of Climate Change." BIS / Banque de France.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

import simulation_engine as sim

US_CARBON_MEAN = 0.60
US_CARBON_STD = 0.12
EU_CARBON_MEAN = 0.30
EU_CARBON_STD = 0.10

GREENWASH_DISCOUNT = 0.10          

GREENWASH_TIER1_COUNT = 30         

BROWN_ASSET_SHARE = 0.20           

GREEN_ASSET_SHARE = 0.15           

DEFAULT_GREEN_SUBSIDY = 0.10       

def assign_climate_exposure(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enrich the bank DataFrame with climate-related columns.

    New columns
    -----------
    carbon_score     :  0–1 intensity   (high = brown / dirty)
    brown_assets     :  $ of fossil-fuel-linked assets
    green_assets     :  $ of renewables / clean-energy assets
    climate_net_pos  :  green_assets − brown_assets  (negative = vulnerable)

    Parameters
    ----------
    df   :  master bank DataFrame (must have 'region', 'total_assets')
    seed :  random seed for reproducibility

    Returns
    -------
    df  (modified in-place & returned)
    """
    rng = np.random.RandomState(seed)
    n = len(df)

    carbon_score = np.zeros(n, dtype=np.float64)
    regions = df['region'].values

    us_mask = (regions == 'US')
    eu_mask = (regions == 'EU')
    other_mask = ~(us_mask | eu_mask)

    carbon_score[us_mask] = rng.normal(US_CARBON_MEAN, US_CARBON_STD,
                                        size=int(us_mask.sum()))
    carbon_score[eu_mask] = rng.normal(EU_CARBON_MEAN, EU_CARBON_STD,
                                        size=int(eu_mask.sum()))
    carbon_score[other_mask] = rng.normal(0.45, 0.15,
                                           size=int(other_mask.sum()))

    tier1_end = min(GREENWASH_TIER1_COUNT, n)
    carbon_score[:tier1_end] -= GREENWASH_DISCOUNT

    carbon_score = np.clip(carbon_score, 0.0, 1.0)

    total_assets = df['total_assets'].values.astype(np.float64)
    brown_assets = total_assets * BROWN_ASSET_SHARE * carbon_score
    green_assets = total_assets * GREEN_ASSET_SHARE * (1.0 - carbon_score)

    df['carbon_score'] = carbon_score
    df['brown_assets'] = brown_assets
    df['green_assets'] = green_assets
    df['climate_net_pos'] = green_assets - brown_assets

    print("\n" + "=" * 60)
    print("CLIMATE EXPOSURE ASSIGNMENT")
    print("=" * 60)
    for region_label, mask in [('US', us_mask), ('EU', eu_mask)]:
        avg_cs = carbon_score[mask].mean() if mask.any() else 0
        tot_brown = brown_assets[mask].sum() / 1e12
        tot_green = green_assets[mask].sum() / 1e12
        print(f"  {region_label}:  avg carbon_score={avg_cs:.2f}  "
              f"brown=${tot_brown:.2f}T  green=${tot_green:.2f}T")
    print(f"  Global Brown: ${brown_assets.sum() / 1e12:.2f}T   "
          f"Green: ${green_assets.sum() / 1e12:.2f}T")

    return df

def run_transition_shock(
    state: dict,
    df: pd.DataFrame,
    carbon_tax_severity: float = 0.50,
    green_subsidy: float = DEFAULT_GREEN_SUBSIDY,
    use_intraday: bool = True,
    n_steps: int = 10,
    uncertainty_sigma: float = 0.05,
    panic_threshold: float = 0.10,
    alpha: float = 0.005,
    margin_sensitivity: float = 1.0,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-5,
    distress_threshold: float = 0.50,
    circuit_breaker_threshold: float = 0.0,
) -> Dict:
    """
    Simulate a Green Swan transition shock and propagate via the network.

    Mechanism
    ---------
    1. Each bank loses:   brown_assets × carbon_tax_severity
    2. Each bank gains:   green_assets × green_subsidy
    3. Net shock applied to external_assets in the state dict.
    4. The most-impacted bank is selected as the trigger.
    5. Existing contagion engine propagates the cascade.

    Parameters
    ----------
    state                : from simulation_engine.compute_state_variables()
    df                   : bank DataFrame (must already have carbon_score etc.)
    carbon_tax_severity  : 0–1  fraction of brown assets destroyed
    green_subsidy        : 0–1  fractional gain on green assets
    use_intraday         : if True, call run_rust_intraday; else run_scenario
    (remaining params forwarded to the simulation engine)

    Returns
    -------
    dict  with all standard simulation results PLUS:
        climate_losses       : per-bank $ brown loss
        climate_gains        : per-bank $ green gain
        climate_net_shock    : per-bank net impact
        total_brown_loss     : aggregate $ destroyed
        total_green_gain     : aggregate $ gained
        aggregate_net_shock  : total net
        carbon_tax_severity  : echo of input
        green_subsidy        : echo of input
    """
    print("\n" + "=" * 60)
    print("GREEN SWAN TRANSITION SHOCK")
    print("=" * 60)

    n = len(df)

    if 'brown_assets' not in df.columns:
        assign_climate_exposure(df)

    brown = df['brown_assets'].values.astype(np.float64)
    green = df['green_assets'].values.astype(np.float64)

    climate_losses = brown * carbon_tax_severity          

    climate_gains = green * green_subsidy                 

    climate_net_shock = climate_losses - climate_gains     

    total_brown_loss = float(climate_losses.sum())
    total_green_gain = float(climate_gains.sum())
    aggregate_net = total_brown_loss - total_green_gain

    print(f"  Carbon Tax Severity: {carbon_tax_severity:.0%}")
    print(f"  Green Subsidy:       {green_subsidy:.0%}")
    print(f"  Brown Losses:        ${total_brown_loss / 1e9:,.1f}B")
    print(f"  Green Gains:         ${total_green_gain / 1e9:,.1f}B")
    print(f"  Aggregate Net Shock: ${aggregate_net / 1e9:,.1f}B")

    shocked_state = {k: (v.copy() if hasattr(v, 'copy') else v)
                     for k, v in state.items()}

    ext = shocked_state['external_assets']
    ext[:] = np.maximum(ext - climate_net_shock, 0.0)

    shocked_state['equity'] = shocked_state['equity'] - climate_net_shock

    original_ext = state['external_assets']
    shock_fraction = np.where(
        original_ext > 0,
        climate_net_shock / original_ext,
        0.0,
    )

    trigger_idx = int(np.argmax(climate_net_shock))
    # FIX: Climate shock is already applied to ALL banks' external_assets
    # above.  Setting trigger_severity = 0 avoids double-dipping the
    # trigger bank.  The clearing / intraday engine will discover defaults
    # organically from the already-weakened state.
    trigger_severity = 0.0

    trigger_name = (str(df.iloc[trigger_idx]['bank_name'])[:40]
                    if pd.notna(df.iloc[trigger_idx].get('bank_name')) else 'Unknown')
    trigger_region = df.iloc[trigger_idx].get('region', '??')

    print(f"\n  Trigger Bank:  {trigger_name}  ({trigger_region})")
    print(f"  Trigger Shock: ${climate_net_shock[trigger_idx] / 1e9:,.1f}B  "
          f"({trigger_severity:.0%} of external assets)")

    worst_idx = np.argsort(climate_net_shock)[::-1][:10]
    print("\n  Top 10 Climate-Impacted Banks:")
    for rank, idx in enumerate(worst_idx):
        bname = str(df.iloc[idx]['bank_name'])[:35]
        bregion = df.iloc[idx]['region']
        loss = climate_net_shock[idx] / 1e9
        cs = df.iloc[idx]['carbon_score']
        print(f"    {rank+1:2d}. {bname:35s} ({bregion})  "
              f"Net=${loss:>8,.1f}B  CS={cs:.2f}")

    if use_intraday:
        results = sim.run_rust_intraday(
            shocked_state, df, trigger_idx, trigger_severity,
            n_steps=n_steps,
            uncertainty_sigma=uncertainty_sigma,
            panic_threshold=panic_threshold,
            alpha=alpha,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            distress_threshold=distress_threshold,
            margin_sensitivity=margin_sensitivity,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )
    else:
        results = sim.run_scenario(
            shocked_state, df, trigger_idx, trigger_severity,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            distress_threshold=distress_threshold,
        )

    results['climate_losses'] = climate_losses
    results['climate_gains'] = climate_gains
    results['climate_net_shock'] = climate_net_shock
    results['total_brown_loss'] = total_brown_loss
    results['total_green_gain'] = total_green_gain
    results['aggregate_net_shock'] = aggregate_net
    results['carbon_tax_severity'] = carbon_tax_severity
    results['green_subsidy'] = green_subsidy
    results['bank_names'] = df['bank_name'].tolist()

    status = results['status']
    for region in ['US', 'EU']:
        rmask = (df['region'].values == region)
        r_def = int(((status == 'Default') & rmask).sum())
        r_dis = int(((status == 'Distressed') & rmask).sum())
        print(f"\n  {region}: {r_def} defaults, {r_dis} distressed")

    print(f"\n  Total Capital Destroyed: ${results['equity_loss'] / 1e12:.2f}T")

    return results

def run_full_climate_scenario(
    W_dense: np.ndarray,
    df: pd.DataFrame,
    carbon_tax_severity: float = 0.50,
    green_subsidy: float = 0.10,
    use_intraday: bool = True,
    **kwargs,
) -> Dict:
    """
    End-to-end helper: assign climate exposure → compute state → shock.

    Parameters
    ----------
    W_dense             :  adjacency matrix (from rescale_matrix_to_dollars)
    df                  :  bank DataFrame
    carbon_tax_severity :  0–1
    green_subsidy       :  0–1
    use_intraday        :  route through intraday engine?
    **kwargs            :  forwarded to run_transition_shock

    Returns  simulation results dict with climate metadata.
    """
    df = assign_climate_exposure(df)
    state = sim.compute_state_variables(W_dense, df)
    return run_transition_shock(
        state, df,
        carbon_tax_severity=carbon_tax_severity,
        green_subsidy=green_subsidy,
        use_intraday=use_intraday,
        **kwargs,
    )

