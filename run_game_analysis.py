"""
run_game_analysis.py — A/B Test: Opaque vs Transparent Information Regimes
==========================================================================

Scenario A  ("Fog of War")      OPAQUE mode  — no reliable public signal.
                                Agents panic, coordination fails, fire-sale cascade.

Scenario B  ("AI Transparency") TRANSPARENT mode — accurate GNN/AI signal.
                                Agents coordinate on fundamentals, runs are averted.

Win Metric:  Total Capital Saved = Loss(A) − Loss(B)
"""

from __future__ import annotations

import numpy as np
from strategic_model import run_game_simulation

def _bar(label: str, value: float, max_val: float, width: int = 40) -> str:
    """Simple ASCII bar for terminal output."""
    filled = int(width * min(value / max(max_val, 1e-12), 1.0))
    return f"  {label}  [{'█' * filled}{'░' * (width - filled)}]  ${value / 1e9:,.2f}B"

def _print_scenario(tag: str, result: dict) -> None:
    """Pretty-print a single scenario result."""
    tl = result['timeline']
    print(f"\n{'─' * 64}")
    print(f"  SCENARIO {tag}  │  Regime: {result['info_regime']}")
    print(f"{'─' * 64}")
    print(f"  Banks: {result['n_banks']}   Steps: {result['n_steps']}   "
          f"True θ: {result['true_solvency']:.2f}   "
          f"r: {result['interest_rate']:.1%}   R: {result['recovery_rate']:.0%}")
    print(f"  Public precision (α): {result['public_precision']:.2f}   "
          f"Private precision (β): {result['private_precision']:.1f}")
    print()
    print(f"  {'Step':>4}  {'Runs':>5}  {'Run%':>6}  {'AvgP(def)':>10}  "
          f"{'U_stay':>7}  {'U_run':>6}  {'Cum Loss':>14}")
    print(f"  {'────':>4}  {'─────':>5}  {'──────':>6}  {'──────────':>10}  "
          f"{'───────':>7}  {'──────':>6}  {'──────────────':>14}")
    for i, t in enumerate(tl['steps']):
        print(
            f"  {t:4d}  {tl['n_runs'][i]:5d}  "
            f"{tl['run_fraction'][i]:5.1%}  "
            f"{tl['avg_belief'][i]:10.4f}  "
            f"{tl['avg_U_stay'][i]:7.4f}  "
            f"{tl['avg_U_run'][i]:6.4f}  "
            f"${tl['cumulative_fire_sale_loss'][i] / 1e9:>12,.2f}B"
        )
    print(f"\n  Total runs:      {result['total_runs']} / {result['total_decisions']}  "
          f"({result['run_rate']:.1%})")
    print(f"  Fire-sale loss:  ${result['total_fire_sale_loss'] / 1e9:,.2f}B")

def run_ab_test(
    n_banks: int = 20,
    n_steps: int = 5,
    true_solvency: float = 0.20,
    interest_rate: float = 0.10,
    recovery_rate: float = 0.40,
    risk_aversion_mean: float = 1.0,
    risk_aversion_std: float = 0.3,
    private_noise_std: float = 0.08,
    initial_exposure_per_bank: float = 1e9,
    fire_sale_haircut: float = 0.20,
    margin_volatility: float = 0.3,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Run both OPAQUE and TRANSPARENT scenarios with identical parameters
    and compare fire-sale losses.

    Returns
    -------
    (result_opaque, result_transparent, capital_saved)
    """
    common = dict(
        n_banks=n_banks,
        n_steps=n_steps,
        true_solvency=true_solvency,
        interest_rate=interest_rate,
        recovery_rate=recovery_rate,
        risk_aversion_mean=risk_aversion_mean,
        risk_aversion_std=risk_aversion_std,
        private_noise_std=private_noise_std,
        initial_exposure_per_bank=initial_exposure_per_bank,
        fire_sale_haircut=fire_sale_haircut,
        margin_volatility=margin_volatility,
        seed=seed,
    )

    result_opaque = run_game_simulation(info_regime="OPAQUE", **common)

    result_transparent = run_game_simulation(info_regime="TRANSPARENT", **common)

    loss_a = result_opaque['total_fire_sale_loss']
    loss_b = result_transparent['total_fire_sale_loss']
    capital_saved = loss_a - loss_b

    if verbose:
        print("\n" + "═" * 64)
        print("  GLOBAL GAMES A/B TEST  —  Morris & Shin (1998)")
        print("═" * 64)

        _print_scenario("A  (Fog of War — OPAQUE)", result_opaque)
        _print_scenario("B  (AI Transparency — TRANSPARENT)", result_transparent)

        max_loss = max(loss_a, loss_b, 1.0)
        print("\n" + "═" * 64)
        print("  COMPARISON")
        print("═" * 64)
        print(_bar("Loss A (Opaque)     ", loss_a, max_loss))
        print(_bar("Loss B (Transparent)", loss_b, max_loss))
        print()
        print(f"  Run-rate  A: {result_opaque['run_rate']:5.1%}   "
              f"B: {result_transparent['run_rate']:5.1%}")
        print()

        print("═" * 64)
        print(f"  ★  TOTAL CAPITAL SAVED  =  ${capital_saved / 1e9:,.2f} BILLION  ★")
        print("═" * 64)

        if capital_saved > 0:
            print("\n  ✓  AI transparency averted the coordination-failure cascade.")
            print("     The accurate public signal anchored expectations,")
            print("     preventing the self-fulfilling bank run.")
        elif capital_saved == 0:
            print("\n  ⚠  No difference — both regimes produced identical outcomes.")
        else:
            print("\n  ✗  Transparent regime performed worse (unexpected).")

    return result_opaque, result_transparent, capital_saved

def sensitivity_sweep(
    solvency_values: list | None = None,
    verbose: bool = True,
) -> list:
    """
    Re-run the A/B test across several true-solvency levels to show
    that the AI advantage is robust across fundamentals.
    """
    if solvency_values is None:
        solvency_values = [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

    rows = []
    for theta in solvency_values:
        res_a, res_b, saved = run_ab_test(
            true_solvency=theta, verbose=False
        )
        rows.append({
            'true_solvency': theta,
            'loss_opaque': res_a['total_fire_sale_loss'],
            'loss_transparent': res_b['total_fire_sale_loss'],
            'capital_saved': saved,
            'run_rate_opaque': res_a['run_rate'],
            'run_rate_transparent': res_b['run_rate'],
        })

    if verbose:
        print("\n" + "═" * 72)
        print("  SENSITIVITY SWEEP  —  Capital Saved vs True Solvency (θ)")
        print("═" * 72)
        print(f"  {'θ':>6}  {'Loss A ($B)':>12}  {'Loss B ($B)':>12}  "
              f"{'Saved ($B)':>11}  {'RunA%':>6}  {'RunB%':>6}")
        print(f"  {'──────':>6}  {'────────────':>12}  {'────────────':>12}  "
              f"{'───────────':>11}  {'──────':>6}  {'──────':>6}")
        for r in rows:
            print(
                f"  {r['true_solvency']:6.2f}  "
                f"{r['loss_opaque'] / 1e9:12,.2f}  "
                f"{r['loss_transparent'] / 1e9:12,.2f}  "
                f"{r['capital_saved'] / 1e9:11,.2f}  "
                f"{r['run_rate_opaque']:5.1%}  "
                f"{r['run_rate_transparent']:5.1%}"
            )
        print()

    return rows

if __name__ == "__main__":
    print("\n" + "█" * 64)
    print("  ENCS — Strategic Interactions Module")
    print("  Morris & Shin (1998) Global Games A/B Test")
    print("█" * 64)

    run_ab_test()

    print("\n")
    sensitivity_sweep()

