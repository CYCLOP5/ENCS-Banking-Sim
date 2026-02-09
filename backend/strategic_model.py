"""
strategic_model.py — Morris & Shin (1998) Global Games Framework
================================================================
Models coordination failures (bank runs) using Bayesian inference on
noisy private and public signals.  Banks decide whether to "Roll Over"
(stay) or "Withdraw" (run) based on expected utility:
    U = E[Return] − λ · Risk
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
    risk_aversion: float            
    total_assets: float
    equity: float
    interbank_exposure: float
    belief_history: List[float] = field(default_factory=list)
    def form_belief(
        self,
        private_signal: float,
        public_signal: float,
        private_precision: float,       
        public_precision: float,        
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
        theta_star = 0.0                       
        posterior_mean = (
            public_precision * public_signal
            + private_precision * private_signal
        ) / (public_precision + private_precision)
        posterior_std = 1.0 / np.sqrt(public_precision + private_precision)
        p_default = float(norm.cdf(
            (theta_star - posterior_mean) / posterior_std
        ))
        self.belief_history.append(p_default)
        return p_default
    def decide(
        self,
        p_default: float,
        interest_rate: float = 0.05,
        recovery_rate: float = 0.40,
        margin_pressure: float = 0.0,
        current_equity: float | None = None,
        initial_equity: float | None = None,
    ) -> Tuple[str, float, float]:
        """
        Expected-utility comparison with **dynamic risk aversion**.
        Payoffs
        -------
        Stay & survive  →  1 + r   (earn coupon)
        Stay & default  →  R       (recovery, R < 1)
        Withdraw (run)  →  1.0     (safe, get principal back)
        Dynamic λ
        ---------
        As a bank's equity depletes, it becomes more risk-averse
        (liquidity hoarding).  If ``current_equity`` and
        ``initial_equity`` are supplied:
            equity_loss_ratio = 1 − current / initial   ∈ [0, 1]
            λ_eff = λ_base · (1 + 2 · equity_loss_ratio)
        When equity is intact  → λ_eff = λ_base  (no change).
        When 50 % lost         → λ_eff = 2 · λ_base  (twice as cautious).
        When fully wiped out   → λ_eff = 3 · λ_base  (maximum hoarding).
        Utility
        -------
        U_stay = E[Return_stay] − λ_eff · σ(Return_stay)
        U_run  = 1.0 + margin_pressure
        margin_pressure captures the liquidity premium when the CCP
        demands variation margin (high volatility → cash is king).
        Returns (decision, U_stay, U_run)
        """
        if current_equity is not None and initial_equity is not None and initial_equity > 0:
            equity_loss_ratio = max(0.0, min(1.0, 1.0 - current_equity / initial_equity))
            effective_lambda = self.risk_aversion * (1.0 + 2.0 * equity_loss_ratio)
        else:
            effective_lambda = self.risk_aversion
        e_return_stay = (
            (1.0 - p_default) * (1.0 + interest_rate)
            + p_default * recovery_rate
        )
        spread = (1.0 + interest_rate) - recovery_rate
        variance_stay = p_default * (1.0 - p_default) * spread ** 2
        risk_stay = np.sqrt(variance_stay)
        U_stay = e_return_stay - effective_lambda * risk_stay
        U_run = 1.0 + margin_pressure
        decision = "WITHDRAW" if U_run > U_stay else "ROLL_OVER"
        return decision, float(U_stay), float(U_run)
@dataclass
class EdgeStrategicAgent:
    """
    Per-edge Bayesian decision maker for the network intraday engine.
    Bank A might trust Bank B but panic about Bank C, so the withdrawal
    decision must be made *per directed edge* (A→B), not per bank.
    Each edge-agent reuses the same Morris & Shin (1998) math that
    :class:`StrategicBankAgent` uses — Bayesian posterior + expected-
    utility comparison — but is parameterised by the specific bilateral
    exposure ``W[lender, borrower]``.
    Attributes
    ----------
    lender_idx   : index of the creditor node in the topology
    borrower_idx : index of the debtor node in the topology
    risk_aversion: λ — drawn once at construction, heterogeneous
    exposure     : current $ value of the directed claim  W[i, j]
    """
    lender_idx:   int
    borrower_idx: int
    risk_aversion: float
    exposure:      float            
    @staticmethod
    def posterior_p_default(
        private_signal: float,
        public_signal: float,
        private_precision: float,
        public_precision: float,
    ) -> float:
        """P(borrower defaults) given noisy private + public signals."""
        theta_star = 0.0
        posterior_mean = (
            public_precision * public_signal
            + private_precision * private_signal
        ) / (public_precision + private_precision)
        posterior_std = 1.0 / np.sqrt(public_precision + private_precision)
        return float(norm.cdf((theta_star - posterior_mean) / posterior_std))
    def decide(
        self,
        p_default: float,
        interest_rate: float = 0.05,
        recovery_rate: float = 0.40,
        margin_pressure: float = 0.0,
        current_equity: float | None = None,
        initial_equity: float | None = None,
    ) -> str:
        """
        Returns ``"WITHDRAW"`` or ``"ROLL_OVER"`` for this specific edge.
        Applies **dynamic risk aversion** — as the *lender's* equity
        depletes, ``λ_eff`` scales up (liquidity hoarding):
            equity_loss_ratio = 1 − current / initial
            λ_eff = λ_base · (1 + 2 · equity_loss_ratio)
        • Stay & survive → 1 + r   (earn coupon on this claim)
        • Stay & default → R       (recovery fraction)
        • Withdraw       → 1.0 + margin_pressure   (safe principal)
        """
        if current_equity is not None and initial_equity is not None and initial_equity > 0:
            eq_loss = max(0.0, min(1.0, 1.0 - current_equity / initial_equity))
            eff_lambda = self.risk_aversion * (1.0 + 2.0 * eq_loss)
        else:
            eff_lambda = self.risk_aversion
        e_return_stay = (
            (1.0 - p_default) * (1.0 + interest_rate)
            + p_default * recovery_rate
        )
        spread = (1.0 + interest_rate) - recovery_rate
        risk_stay = np.sqrt(p_default * (1.0 - p_default) * spread ** 2)
        U_stay = e_return_stay - eff_lambda * risk_stay
        U_run  = 1.0 + margin_pressure
        return "WITHDRAW" if U_run > U_stay else "ROLL_OVER"
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
    private_precision = 1.0 / (private_noise_std ** 2)     
    if info_regime == "TRANSPARENT":
        public_precision = 100.0                            
        public_signal = true_solvency + rng.normal(0, 0.01)
    elif info_regime == "OPAQUE":
        public_precision = 0.01                             
        public_signal = 0.0                                 
    else:
        raise ValueError(
            f"Unknown info_regime '{info_regime}'. Use 'OPAQUE' or 'TRANSPARENT'."
        )
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
    for t in range(1, n_steps + 1):
        step_beliefs: List[float] = []
        step_U_stay: List[float] = []
        step_U_run: List[float] = []
        step_decisions: List[str] = []
        loss_fraction = cumulative_loss / max(
            remaining_exposure + cumulative_loss, 1.0
        )
        margin_pressure = margin_volatility * (0.02 + 0.10 * loss_fraction)
        n_runs_t = 0
        n_rollovers_t = 0
        for agent in agents:
            effective_solvency = true_solvency * (1.0 - 0.5 * loss_fraction)
            private_signal = effective_solvency + rng.normal(0, private_noise_std)
            p_default = agent.form_belief(
                private_signal=private_signal,
                public_signal=public_signal,
                private_precision=private_precision,
                public_precision=public_precision,
            )
            agent_current_equity = agent.equity * (1.0 - loss_fraction)
            decision, U_stay, U_run = agent.decide(
                p_default=p_default,
                interest_rate=interest_rate,
                recovery_rate=recovery_rate,
                margin_pressure=margin_pressure,
                current_equity=agent_current_equity,
                initial_equity=agent.equity,
            )
            step_decisions.append(decision)
            step_beliefs.append(p_default)
            step_U_stay.append(U_stay)
            step_U_run.append(U_run)
            if decision == "WITHDRAW":
                n_runs_t += 1
            else:
                n_rollovers_t += 1
        if remaining_exposure > 0:
            withdrawal_volume = (n_runs_t / n_banks) * remaining_exposure
            step_loss = withdrawal_volume * fire_sale_haircut
            cumulative_loss += step_loss
            remaining_exposure -= withdrawal_volume
            remaining_exposure = max(remaining_exposure, 0.0)
        else:
            step_loss = 0.0
        run_frac = n_runs_t / n_banks
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