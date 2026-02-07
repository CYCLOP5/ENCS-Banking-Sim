"""
strategic_model.py — Morris & Shin (1998) Global Games Framework
================================================================

Models coordination failures (bank runs) using Bayesian inference on
noisy private and public signals.  Banks decide whether to "Roll Over"
(stay) or "Withdraw" (run) based on expected utility:

    U = E[Return] − λ · Risk

Key insight: banks can fail not just from insolvency, but from
*coordination failure* — self-fulfilling panics where each agent
withdraws because it fears others will withdraw first.

Regime Switch:
    OPAQUE      — public signal is uninformative → Prisoner's Dilemma dynamics
    TRANSPARENT — public signal is accurate (from GNN/AI) → coordination success

Reference
---------
Morris, S. & Shin, H.S. (1998). "Unique Equilibrium in a Model of
Self-Fulfilling Currency Attacks."  American Economic Review, 88(3).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class StrategicBankAgent:
    """
    A "Smart Bank" that maximises  U = E[Return] − λ · Risk.

    Implements Bayesian belief updating from the Morris & Shin (1998)
    global-games framework with heterogeneous private signals and a
    common public signal.

    Attributes
    ----------
    bank_id            : unique identifier
    name               : human-readable label
    risk_aversion      : λ  — higher → more conservative
    total_assets       : balance-sheet total assets ($)
    equity             : equity capital ($)
    interbank_exposure : how much this agent has lent to the target ($)
    belief_history     : posterior P(default) at each time step
    """

    bank_id: str
    name: str
    risk_aversion: float            # λ
    total_assets: float
    equity: float
    interbank_exposure: float
    belief_history: List[float] = field(default_factory=list)

    # ---- Bayesian belief formation ----------------------------------------

    def form_belief(
        self,
        private_signal: float,
        public_signal: float,
        private_precision: float,       # β = 1 / σ²_private
        public_precision: float,        # α = 1 / σ²_public
    ) -> float:
        """
        Bayesian posterior probability of default.

        Model
        -----
        True state  θ  ~ N(y, 1/α)          y = public signal
        Private obs xᵢ = θ + εᵢ,  εᵢ ~ N(0, 1/β)

        Posterior:  θ | xᵢ, y  ~  N(μ_post, σ²_post)

            μ_post  = (α·y + β·xᵢ) / (α + β)
            σ²_post = 1 / (α + β)

        Default threshold θ* = 0  (equity ≤ 0 → insolvent).

        Returns  P(θ < θ*) = Φ( (θ* − μ_post) / σ_post )
        """
        theta_star = 0.0                       # insolvency boundary

        # Precision-weighted posterior mean
        posterior_mean = (
            public_precision * public_signal
            + private_precision * private_signal
        ) / (public_precision + private_precision)

        # Posterior standard deviation
        posterior_std = 1.0 / np.sqrt(public_precision + private_precision)

        # P(default) via standard normal CDF
        p_default = float(norm.cdf(
            (theta_star - posterior_mean) / posterior_std
        ))

        self.belief_history.append(p_default)
        return p_default

    # ---- Expected-utility decision ----------------------------------------

    def decide(
        self,
        p_default: float,
        interest_rate: float = 0.05,
        recovery_rate: float = 0.40,
        margin_pressure: float = 0.0,
    ) -> Tuple[str, float, float]:
        """
        Expected-utility comparison:

        Payoffs
        -------
        Stay & survive  →  1 + r   (earn coupon)
        Stay & default  →  R       (recovery, R < 1)
        Withdraw (run)  →  1.0     (safe, get principal back)

        Utility
        -------
        U_stay = E[Return_stay] − λ · σ(Return_stay)
        U_run  = 1.0 + margin_pressure

        margin_pressure captures the liquidity premium when the CCP
        demands variation margin (high volatility → cash is king).

        Returns (decision, U_stay, U_run)
        """
        # Expected return from rolling over
        e_return_stay = (
            (1.0 - p_default) * (1.0 + interest_rate)
            + p_default * recovery_rate
        )

        # Standard-deviation of the binary-outcome return
        spread = (1.0 + interest_rate) - recovery_rate
        variance_stay = p_default * (1.0 - p_default) * spread ** 2
        risk_stay = np.sqrt(variance_stay)

        # Risk-adjusted utility
        U_stay = e_return_stay - self.risk_aversion * risk_stay

        # Safe exit  +  liquidity premium from margin pressure
        U_run = 1.0 + margin_pressure

        decision = "WITHDRAW" if U_run > U_stay else "ROLL_OVER"
        return decision, float(U_stay), float(U_run)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_game_simulation(
    n_banks: int = 20,
    n_steps: int = 5,
    info_regime: str = "OPAQUE",
    true_solvency: float = 0.20,
    interest_rate: float = 0.10,
    recovery_rate: float = 0.40,
    risk_aversion_mean: float = 1.0,
    risk_aversion_std: float = 0.3,
    private_noise_std: float = 0.08,
    initial_exposure_per_bank: float = 1e9,
    fire_sale_haircut: float = 0.20,
    margin_volatility: float = 0.0,
    seed: int = 42,
) -> Dict:
    """
    Morris & Shin (1998) global-games coordination simulation.

    At each step every agent:
        1. Receives a noisy *private* signal about the target's solvency.
        2. Observes a *public* signal (quality depends on ``info_regime``).
        3. Forms a Bayesian posterior P(default).
        4. Computes expected utility and decides ROLL_OVER or WITHDRAW.

    Aggregate withdrawals trigger proportional fire-sale losses, which
    feed back into agents' signals at the next step (self-fulfilling).

    Parameters
    ----------
    n_banks                   : number of creditor agents
    n_steps                   : discrete time periods  (t = 1 … T)
    info_regime               : "OPAQUE" or "TRANSPARENT"
    true_solvency             : θ — hidden fundamental solvency ratio
    interest_rate              : r — coupon on interbank lending
    recovery_rate              : R — cents-on-the-dollar if target defaults
    risk_aversion_mean / _std  : distribution of λ across agents
    private_noise_std          : σ_ε  — noise on each agent's private signal
    initial_exposure_per_bank  : $ lent per agent to the target
    fire_sale_haircut          : fraction of withdrawal volume lost to fire sales
    margin_volatility          : drives CCP margin pressure (0 = calm, 1 = crisis)
    seed                       : random seed

    Returns
    -------
    dict  with keys:
        timeline      — per-step arrays (runs, losses, beliefs, …)
        total_fire_sale_loss
        run_rate      — aggregate fraction of WITHDRAW decisions
        agents        — list of StrategicBankAgent objects (with belief_history)
    """
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 1.  Configure regime-dependent signal precision
    # ------------------------------------------------------------------
    private_precision = 1.0 / (private_noise_std ** 2)     # β

    if info_regime == "TRANSPARENT":
        # AI / GNN provides a near-perfect public signal
        public_precision = 100.0                            # α  (very high)
        public_signal = true_solvency + rng.normal(0, 0.01)
    elif info_regime == "OPAQUE":
        # Public signal ≈ noise  →  agents fall back on private info
        public_precision = 0.01                             # α ≈ 0
        public_signal = 0.0                                 # uninformative
    else:
        raise ValueError(
            f"Unknown info_regime '{info_regime}'. Use 'OPAQUE' or 'TRANSPARENT'."
        )

    # ------------------------------------------------------------------
    # 2.  Create heterogeneous agents
    # ------------------------------------------------------------------
    agents: List[StrategicBankAgent] = []
    for i in range(n_banks):
        lam = max(0.1, rng.normal(risk_aversion_mean, risk_aversion_std))
        agents.append(StrategicBankAgent(
            bank_id=f"BANK_{i:03d}",
            name=f"Strategic Bank {i}",
            risk_aversion=lam,
            total_assets=initial_exposure_per_bank * 10,
            equity=initial_exposure_per_bank * 1.5,
            interbank_exposure=initial_exposure_per_bank,
        ))

    # ------------------------------------------------------------------
    # 3.  Timeline accumulators
    # ------------------------------------------------------------------
    timeline: Dict[str, list] = {
        'steps': [],
        'n_runs': [],
        'n_rollovers': [],
        'run_fraction': [],
        'cumulative_fire_sale_loss': [],
        'step_fire_sale_loss': [],
        'avg_belief': [],
        'avg_U_stay': [],
        'avg_U_run': [],
        'margin_pressure': [],
        'decisions': [],
    }

    remaining_exposure = float(n_banks * initial_exposure_per_bank)
    cumulative_loss = 0.0

    # ------------------------------------------------------------------
    # 4.  Main loop
    # ------------------------------------------------------------------
    for t in range(1, n_steps + 1):

        step_beliefs: List[float] = []
        step_U_stay: List[float] = []
        step_U_run: List[float] = []
        step_decisions: List[str] = []

        # Feedback: how much of the exposure pool has been destroyed?
        loss_fraction = cumulative_loss / max(
            remaining_exposure + cumulative_loss, 1.0
        )

        # CCP margin pressure — rises with both market-wide volatility
        # and accumulated fire-sale damage
        margin_pressure = margin_volatility * (0.02 + 0.10 * loss_fraction)

        n_runs_t = 0
        n_rollovers_t = 0

        for agent in agents:
            # Effective solvency degrades as fire-sale losses build
            effective_solvency = true_solvency * (1.0 - 0.5 * loss_fraction)

            # Private signal = noisy observation of true state
            private_signal = effective_solvency + rng.normal(0, private_noise_std)

            # Bayesian update
            p_default = agent.form_belief(
                private_signal=private_signal,
                public_signal=public_signal,
                private_precision=private_precision,
                public_precision=public_precision,
            )

            # Strategic decision
            decision, U_stay, U_run = agent.decide(
                p_default=p_default,
                interest_rate=interest_rate,
                recovery_rate=recovery_rate,
                margin_pressure=margin_pressure,
            )

            step_decisions.append(decision)
            step_beliefs.append(p_default)
            step_U_stay.append(U_stay)
            step_U_run.append(U_run)

            if decision == "WITHDRAW":
                n_runs_t += 1
            else:
                n_rollovers_t += 1

        # ---- Fire-sale losses from aggregate withdrawals ----------------
        if remaining_exposure > 0:
            withdrawal_volume = (n_runs_t / n_banks) * remaining_exposure
            step_loss = withdrawal_volume * fire_sale_haircut
            cumulative_loss += step_loss
            remaining_exposure -= withdrawal_volume
            remaining_exposure = max(remaining_exposure, 0.0)
        else:
            step_loss = 0.0

        run_frac = n_runs_t / n_banks

        # ---- Record step ------------------------------------------------
        timeline['steps'].append(t)
        timeline['n_runs'].append(n_runs_t)
        timeline['n_rollovers'].append(n_rollovers_t)
        timeline['run_fraction'].append(run_frac)
        timeline['cumulative_fire_sale_loss'].append(cumulative_loss)
        timeline['step_fire_sale_loss'].append(step_loss)
        timeline['avg_belief'].append(float(np.mean(step_beliefs)))
        timeline['avg_U_stay'].append(float(np.mean(step_U_stay)))
        timeline['avg_U_run'].append(float(np.mean(step_U_run)))
        timeline['margin_pressure'].append(margin_pressure)
        timeline['decisions'].append(step_decisions)

    # ------------------------------------------------------------------
    # 5.  Summary
    # ------------------------------------------------------------------
    total_runs = sum(timeline['n_runs'])
    total_decisions = n_banks * n_steps

    return {
        'info_regime': info_regime,
        'n_banks': n_banks,
        'n_steps': n_steps,
        'true_solvency': true_solvency,
        'interest_rate': interest_rate,
        'recovery_rate': recovery_rate,
        'risk_aversion_mean': risk_aversion_mean,
        'private_noise_std': private_noise_std,
        'public_precision': public_precision,
        'private_precision': private_precision,
        'public_signal': float(public_signal),
        'total_runs': total_runs,
        'total_decisions': total_decisions,
        'run_rate': total_runs / max(total_decisions, 1),
        'total_fire_sale_loss': cumulative_loss,
        'timeline': timeline,
        'agents': agents,
    }
