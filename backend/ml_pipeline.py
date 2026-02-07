"""
ml_pipeline.py — Edge-Aware PNA GNN for Systemic Risk Classification
=====================================================================

Architecture: SystRiskPNA  (Principal Neighbourhood Aggregation)
  • 3× PNAConv layers with edge_attr (exposure magnitude + concentration)
  • Residual connections + LayerNorm + Dropout
  • Node-level binary classification: Safe (0) vs Risky (1)
  • Explainability via PyG GNNExplainer
  • MLflow experiment tracking

Usage:
    python ml_pipeline.py                        # full pipeline
    python ml_pipeline.py --generate-only        # data generation only
    python ml_pipeline.py --train-only           # train on existing data
    python ml_pipeline.py --runs 500             # custom run count
    python ml_pipeline.py --explain              # run explainability after training
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import simulation_engine as sim

# ---------------------------------------------------------------------------
# Workaround: some PyTorch builds require Intel ITT JIT profiling symbols
# (e.g., iJIT_NotifyEvent) at dynamic link time.  If a stub shared library
# exists next to this script, preload it with RTLD_GLOBAL *before* the first
# `import torch` so the symbol is already available when libtorch_cpu.so loads.
# ---------------------------------------------------------------------------
import ctypes as _ctypes

_itt_stub = os.environ.get(
    "ENCS_ITT_STUB_PATH",
    str(Path(__file__).parent / "libittnotify_stub.so"),
)
if os.path.isfile(_itt_stub):
    _ctypes.CDLL(_itt_stub, mode=getattr(_ctypes, "RTLD_GLOBAL", 0))

import torch  # noqa: E402  (must come after the stub preload)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import PNAConv, global_mean_pool
from torch_geometric.utils import degree

try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "output"
MODEL_PATH = BASE_DIR / "gnn_model.pth"
DATASET_PATH = DATA_DIR / "gnn_dataset.pt"

# ── Number of edge features produced by build_edge_attr ───────────────────
EDGE_FEAT_DIM = 3   # [log1p(w), share_of_obligor, share_of_creditor]
NODE_FEAT_DIM = 7    # from build_node_features

# Default edge pruning to keep training feasible on GPU.
# You can override at runtime with ENCS_PRUNE_TOPK or --prune-topk.
PRUNE_TOPK_DEFAULT = int(os.environ.get("ENCS_PRUNE_TOPK", "64"))


# ===========================================================================
#  Feature Engineering
# ===========================================================================

def _build_node_features_from_strength(
    out_strength: np.ndarray, in_strength: np.ndarray, df: pd.DataFrame
) -> np.ndarray:
    n = len(df)

    ta = df["total_assets"].values.astype(np.float64)
    lev = (
        df["leverage_ratio"].values.astype(np.float64)
        if "leverage_ratio" in df.columns
        else np.zeros(n)
    )
    ib_ratio = (
        df["interbank_ratio"].values.astype(np.float64)
        if "interbank_ratio" in df.columns
        else np.full(n, 0.2)
    )
    eq = (
        df["equity_capital"].values.astype(np.float64)
        if "equity_capital" in df.columns
        else np.maximum(ta - df["total_liabilities"].values, 0)
    )
    deriv = (
        df["deriv_ir_notional"].fillna(0).values.astype(np.float64)
        if "deriv_ir_notional" in df.columns
        else np.zeros(n)
    )

    features = np.column_stack(
        [
            np.log1p(np.abs(ta)),
            np.clip(lev, 0, 100),
            np.log1p(out_strength),
            np.log1p(in_strength),
            ib_ratio,
            np.log1p(np.abs(eq)),
            np.log1p(deriv),
        ]
    )
    return features.astype(np.float32)


def build_node_features(W: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Construct per-node feature vectors from the network topology + balance sheet.

    Features (7-dim):
        0. log(total_assets)            — size
        1. leverage_ratio               — fragility
        2. log(out_strength + 1)        — how much node owes (contagion source)
        3. log(in_strength + 1)         — how much node is owed (contagion sink)
        4. interbank_ratio              — exposure concentration
        5. log(equity_capital + 1)      — buffer
        6. log(deriv_ir_notional + 1)   — derivatives exposure
    """
    out_strength = W.sum(axis=1)
    in_strength = W.sum(axis=0)
    return _build_node_features_from_strength(out_strength, in_strength, df)


def build_edge_index(W: np.ndarray, topk_per_row: int | None = None) -> np.ndarray:
    """Convert dense adjacency to COO edge_index (2 × E).

    If `topk_per_row` is set, keeps only the top-K outgoing edges per node.
    This is critical for GPU memory when the graph is extremely dense.
    """
    if topk_per_row is None:
        rows, cols = np.nonzero(W)
        return np.stack([rows, cols], axis=0).astype(np.int64)

    n = W.shape[0]
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for i in range(n):
        nz = np.nonzero(W[i])[0]
        if nz.size == 0:
            continue
        if nz.size > topk_per_row:
            vals = W[i, nz]
            keep = nz[np.argpartition(vals, -topk_per_row)[-topk_per_row:]]
            nz = keep
        rows_list.append(np.full(nz.size, i, dtype=np.int64))
        cols_list.append(nz.astype(np.int64))

    if not rows_list:
        return np.zeros((2, 0), dtype=np.int64)
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    return np.stack([rows, cols], axis=0).astype(np.int64)


def _build_edge_attr_from_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    out_strength: np.ndarray,
    in_strength: np.ndarray,
) -> np.ndarray:
    eps = 1e-12
    e0 = np.log1p(weights)
    e1 = weights / (out_strength[rows] + eps)
    e2 = weights / (in_strength[cols] + eps)
    return np.column_stack([e0, e1, e2]).astype(np.float32)


def build_edge_attr(W: np.ndarray, edge_index: np.ndarray | None = None) -> np.ndarray:
    """
    Build per-edge feature vectors from the dollar-weighted adjacency matrix.

    For every edge (i → j) with weight W[i,j]:
        e0 = log1p(W[i,j])                           — exposure magnitude
        e1 = W[i,j] / (sum_k W[i,k] + eps)           — share of obligor i's total debt
        e2 = W[i,j] / (sum_k W[k,j] + eps)           — share of creditor j's incoming claims

    Returns: np.ndarray of shape (E, 3), dtype float32
    """
    if edge_index is None:
        rows, cols = np.nonzero(W)
    else:
        rows = edge_index[0]
        cols = edge_index[1]

    weights = W[rows, cols].astype(np.float64)
    out_strength = W.sum(axis=1)
    in_strength = W.sum(axis=0)
    return _build_edge_attr_from_edges(rows, cols, weights, out_strength, in_strength)


def status_to_label(status_array: np.ndarray) -> np.ndarray:
    """Map status strings -> binary labels.  1 = Risky (Default or Distressed), 0 = Safe.

    Robust to case variations and encoding differences from Rust interop.
    """
    labels = np.zeros(len(status_array), dtype=np.int64)

    normalised = np.array(
        [
            s.lower().strip() if isinstance(s, str) else str(s).lower().strip()
            for s in status_array
        ]
    )

    labels[normalised == "default"] = 1
    labels[normalised == "distressed"] = 1

    n_risky = (labels == 1).sum()
    if n_risky == 0 and len(labels) > 0:
        raise ValueError(
            f"No risky labels found in status array! "
            f"Unique inputs: {np.unique(status_array)}  "
            f"Unique normalised: {np.unique(normalised)}"
        )

    return labels


# ===========================================================================
#  Multiprocessing — Monte Carlo data generation
# ===========================================================================

_SHARED: dict = {}


def _mp_worker_init():
    """Limit per-worker OpenMP / MKL threads to avoid contention."""
    torch.set_num_threads(1)


def _single_mc_run(args):
    """Execute one Monte Carlo simulation run (for multiprocessing.Pool)."""
    run_idx, noise_pct, n_steps, seed_base = args

    W_base = _SHARED["W_base"]
    df = _SHARED["df"]
    edge_rows = _SHARED["edge_rows"]
    edge_cols = _SHARED["edge_cols"]
    base_edge_weights = _SHARED["base_edge_weights"]
    pool_top30 = _SHARED["pool_top30"]
    pool_mid = _SHARED["pool_mid"]
    pool_small = _SHARED["pool_small"]
    pool_all = _SHARED["pool_all"]

    rng = np.random.RandomState(seed=seed_base + run_idx)

    # ── Determine regime ──────────────────────────────────────────────
    r = rng.random()
    if r < 0.35:
        regime = "calm"
    elif r < 0.70:
        regime = "moderate"
    else:
        regime = "stressed"

    # ── Perturb exposures ONLY on existing edges (saves huge memory) ─
    edge_noise = (1.0 + rng.uniform(-noise_pct, noise_pct, size=base_edge_weights.shape)).astype(
        np.float32
    )
    edge_weights = (base_edge_weights * edge_noise).astype(np.float64)

    # Strengths for node features / edge attributes
    n = W_base.shape[0]
    out_strength = np.bincount(edge_rows, weights=edge_weights, minlength=n)
    in_strength = np.bincount(edge_cols, weights=edge_weights, minlength=n)

    # ── Regime-specific parameters ────────────────────────────────────
    if regime == "calm":
        trigger_idx = int(rng.choice(pool_small if len(pool_small) > 0 else pool_all))
        severity = float(rng.uniform(0.01, 0.10))
        sigma = rng.uniform(0.01, 0.03)
        panic_th = rng.uniform(0.30, 0.50)
        alpha = rng.uniform(0.0005, 0.002)
        margin_sens = 0.0
    elif regime == "moderate":
        trigger_idx = int(rng.choice(pool_mid if len(pool_mid) > 0 else pool_all))
        severity = float(rng.uniform(0.05, 0.40))
        sigma = rng.uniform(0.03, 0.06)
        panic_th = rng.uniform(0.15, 0.30)
        alpha = rng.uniform(0.001, 0.004)
        margin_sens = rng.uniform(0.0, 0.5)
    else:  # stressed
        trigger_idx = int(rng.choice(pool_top30))
        severity = float(rng.uniform(0.30, 1.0))
        sigma = rng.uniform(0.05, 0.10)
        panic_th = rng.uniform(0.05, 0.15)
        alpha = rng.uniform(0.004, 0.010)
        margin_sens = rng.uniform(0.5, 2.0)

    # ── Run simulation ────────────────────────────────────────────────
    # The Rust engine expects a full dense matrix for state variables.
    # Build it here (still cheaper than sending huge tensors over IPC).
    W_noisy = np.zeros_like(W_base, dtype=np.float64)
    W_noisy[edge_rows, edge_cols] = edge_weights
    np.fill_diagonal(W_noisy, 0.0)

    state = sim.compute_state_variables(W_noisy, df)

    try:
        results = sim.run_rust_intraday(
            state,
            df,
            trigger_idx,
            severity,
            n_steps=n_steps,
            uncertainty_sigma=sigma,
            panic_threshold=panic_th,
            alpha=alpha,
            margin_sensitivity=margin_sens,
            distress_threshold=0.5,
        )
    except Exception as exc:
        if run_idx < 5:
            traceback.print_exc()
        else:
            print(
                f"  ⚠  MC run {run_idx} failed: {type(exc).__name__}: "
                f"{str(exc)[:120]}"
            )
        return None

    # ── Build lightweight arrays (avoid pickling full PyG Data over IPC) ──
    node_feats = _build_node_features_from_strength(out_strength, in_strength, df)

    try:
        labels = status_to_label(results["status"])  # (N,) int64
    except ValueError:
        return None

    # Return only the parts that vary per run (node feats + labels).
    # edge_index is identical across runs (same topology); edge_attr is
    # rebuilt from W_noisy per-run but is huge (1.3 M × 3).  Instead we
    # return the *noise multiplier* so the main process can reconstruct.
    return (node_feats, labels, edge_noise, regime)


def generate_dataset(
    n_runs: int = 500,
    noise_pct: float = 0.10,
    n_steps: int = 5,
    n_workers: int | None = None,
    verbose: bool = True,
    prune_topk: int | None = PRUNE_TOPK_DEFAULT,
) -> list:
    """
    Run the simulation `n_runs` times with random perturbations.
    Returns a list of PyG Data objects (each with x, edge_index, edge_attr, y).
    """
    global _SHARED

    print("\n" + "=" * 60)
    print(f"GNN DATA GENERATION — {n_runs} Monte Carlo runs  (3 regimes)")
    print("=" * 60)

    W_sparse, df = sim.load_and_align_network()
    W_base = sim.rescale_matrix_to_dollars(W_sparse, df)
    n = W_base.shape[0]

    if prune_topk is not None and prune_topk > 0:
        print(f"  Edge pruning: keeping top-{prune_topk} outgoing edges per node")
    edge_index_base = build_edge_index(W_base, topk_per_row=prune_topk)
    edge_rows = edge_index_base[0].astype(np.int64)
    edge_cols = edge_index_base[1].astype(np.int64)
    base_edge_weights = W_base[edge_rows, edge_cols].astype(np.float32)

    out_strength = W_base.sum(axis=1)
    rank = np.argsort(out_strength)[::-1]
    pool_top30 = rank[: min(30, n)]
    pool_mid = rank[min(30, n) : min(200, n)]
    pool_small = rank[min(200, n) :]
    pool_all = np.arange(n)

    _SHARED.update(
        {
            "W_base": W_base,
            "df": df,
            "edge_rows": edge_rows,
            "edge_cols": edge_cols,
            "base_edge_weights": base_edge_weights,
            "pool_top30": pool_top30,
            "pool_mid": pool_mid,
            "pool_small": pool_small,
            "pool_all": pool_all,
        }
    )

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 12)
    n_workers = max(1, n_workers)

    args_list = [(i, noise_pct, n_steps, 42) for i in range(n_runs)]
    chunksize = max(1, n_runs // (n_workers * 8))

    dataset: list = []
    label_counts = {0: 0, 1: 0}
    regime_counts = {"calm": 0, "moderate": 0, "stressed": 0}
    failed = 0
    t0 = time.time()

    print(f"  Workers: {n_workers}  |  Chunksize: {chunksize}")

    # Pre-compute shared tensors (identical across all runs)
    edge_index_t = torch.tensor(edge_index_base, dtype=torch.long)

    with mp.Pool(processes=n_workers, initializer=_mp_worker_init) as pool:
        results_iter = pool.imap_unordered(
            _single_mc_run, args_list, chunksize=chunksize
        )
        for i, result in enumerate(results_iter):
            if result is not None:
                node_feats, labels, edge_noise, regime = result

                # Reconstruct edge weights and edge_attr in main process
                edge_weights = (_SHARED["base_edge_weights"].astype(np.float64) * edge_noise.astype(np.float64))
                out_strength = np.bincount(edge_rows, weights=edge_weights, minlength=n)
                in_strength = np.bincount(edge_cols, weights=edge_weights, minlength=n)
                edge_attr_np = _build_edge_attr_from_edges(
                    edge_rows,
                    edge_cols,
                    edge_weights,
                    out_strength,
                    in_strength,
                )

                data = Data(
                    x=torch.tensor(node_feats, dtype=torch.float32),
                    edge_index=edge_index_t,   # shared reference
                    # Store edge_attr as fp16 to drastically reduce RAM/disk.
                    edge_attr=torch.tensor(edge_attr_np, dtype=torch.float16),
                    y=torch.tensor(labels, dtype=torch.long),
                )
                dataset.append(data)
                regime_counts[regime] += 1
                for lbl in labels:
                    label_counts[int(lbl)] += 1
            else:
                failed += 1

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_runs - i - 1) / rate
                frac_risky = (
                    label_counts[1] / max(label_counts[0] + label_counts[1], 1) * 100
                )
                print(
                    f"  [{i + 1}/{n_runs}] {rate:.1f} runs/sec | ETA {eta:.0f}s | "
                    f"Safe={label_counts[0]:,} Risky={label_counts[1]:,} ({frac_risky:.0f}%)"
                    f" | calm={regime_counts['calm']} mod={regime_counts['moderate']}"
                    f" stress={regime_counts['stressed']}"
                )

    elapsed = time.time() - t0
    print(f"\n  Generated {len(dataset)} graphs in {elapsed:.1f}s  ({failed} failed)")
    print(
        f"  Regimes: calm={regime_counts['calm']}, moderate={regime_counts['moderate']}, "
        f"stressed={regime_counts['stressed']}"
    )
    print(f"  Label distribution: Safe={label_counts[0]:,}  Risky={label_counts[1]:,}")
    total = label_counts[0] + label_counts[1]
    if total > 0:
        risky_frac = label_counts[1] / total * 100
        print(f"  Class balance: {risky_frac:.1f}% risky")
        if risky_frac < 5.0:
            print(
                "  ⚠  WARNING: Risky class is <5% of data. "
                "Consider adjusting regime parameters or adding more stressed runs."
            )
    if regime_counts.get("stressed", 0) == 0:
        print("  ⚠  WARNING: No stressed-regime runs succeeded!")

    return dataset


# ===========================================================================
#  Dataset Aggregation — collapse MC runs into a single graph
# ===========================================================================

def aggregate_dataset(dataset: list) -> Data:
    """
    Collapse N Monte-Carlo graph snapshots into a *single* graph
    where y[i] = fraction of MC runs in which node i was Risky.

    Also enriches node features with topology-derived features and
    z-score standardises everything so the GNN sees unit-variance inputs.
    """
    print("\n" + "=" * 60)
    print(f"AGGREGATING {len(dataset)} MC GRAPHS → SINGLE RISK-FREQUENCY GRAPH")
    print("=" * 60)

    n_nodes = dataset[0].num_nodes
    n_runs = len(dataset)

    # Stack labels: (n_runs, n_nodes) → mean → (n_nodes,) risk frequency
    labels = torch.stack([d.y for d in dataset]).float()       # (R, N)
    risk_freq = labels.mean(dim=0)                              # (N,)

    # Use mean features across runs (they barely vary anyway)
    feats = torch.stack([d.x for d in dataset]).mean(dim=0)    # (N, F)

    # Use mean edge_attr across runs
    edge_attrs = torch.stack([d.edge_attr.float() for d in dataset]).mean(dim=0)

    # Topology is identical; take from first graph
    edge_index = dataset[0].edge_index

    # ── Enrich with topology-derived features ─────────────────────────
    from torch_geometric.utils import degree as _deg
    ei_np = edge_index.numpy()
    in_deg = _deg(edge_index[1], num_nodes=n_nodes, dtype=torch.float).unsqueeze(1)   # (N,1)
    out_deg = _deg(edge_index[0], num_nodes=n_nodes, dtype=torch.float).unsqueeze(1)  # (N,1)

    # Net strength = out_strength - in_strength (from raw features cols 2,3)
    out_str = feats[:, 2:3]   # log_out_strength
    in_str = feats[:, 3:4]    # log_in_strength
    net_str = out_str - in_str                                   # (N,1)

    # Strength ratio = out / (out + in + eps)
    str_ratio = out_str / (out_str + in_str + 1e-6)             # (N,1)

    # Asset-to-equity ratio (from cols 0 and 5)
    log_ta = feats[:, 0:1]
    log_eq = feats[:, 5:6]
    asset_equity = log_ta - log_eq                               # (N,1) log(TA/E)

    # Degree ratio
    deg_ratio = out_deg / (out_deg + in_deg + 1e-6)             # (N,1)

    # Concatenate: original 7 + 6 new = 13 features
    feats_enriched = torch.cat([
        feats,          # 7 original
        torch.log1p(in_deg),       # 8: log in-degree
        torch.log1p(out_deg),      # 9: log out-degree
        net_str,        # 10: net strength
        str_ratio,      # 11: strength ratio
        asset_equity,   # 12: asset-to-equity (log-scale)
        deg_ratio,      # 13: degree ratio
    ], dim=1)

    # ── Z-score standardisation ───────────────────────────────────────
    feat_mean = feats_enriched.mean(dim=0, keepdim=True)
    feat_std = feats_enriched.std(dim=0, keepdim=True).clamp(min=1e-6)
    feats_norm = (feats_enriched - feat_mean) / feat_std

    # Also standardise edge features
    ea_mean = edge_attrs.mean(dim=0, keepdim=True)
    ea_std = edge_attrs.std(dim=0, keepdim=True).clamp(min=1e-6)
    edge_attrs_norm = (edge_attrs - ea_mean) / ea_std

    agg = Data(
        x=feats_norm,
        edge_index=edge_index,
        edge_attr=edge_attrs_norm,
        y=risk_freq,                    # continuous in [0, 1]
    )
    # Store normalization stats for inference
    agg.feat_mean = feat_mean.squeeze(0)
    agg.feat_std = feat_std.squeeze(0)
    agg.ea_mean = ea_mean.squeeze(0)
    agg.ea_std = ea_std.squeeze(0)

    n_feat = feats_norm.shape[1]
    print(f"  Nodes: {n_nodes}  |  Edges: {edge_index.shape[1]}  |  Features: {n_feat}")
    print(f"  Risk frequency — min: {risk_freq.min():.4f}  "
          f"max: {risk_freq.max():.4f}  mean: {risk_freq.mean():.4f}  "
          f"std: {risk_freq.std():.4f}")
    print(f"  Nodes with risk_freq > 0.50: {(risk_freq > 0.50).sum().item()}")
    print(f"  Nodes with risk_freq > 0.25: {(risk_freq > 0.25).sum().item()}")
    print(f"  Nodes with risk_freq < 0.05: {(risk_freq < 0.05).sum().item()}")
    return agg


# ===========================================================================
#  Model Definition — Edge-Aware PNA
# ===========================================================================

class SystRiskPNA(nn.Module):
    """
    Edge-aware Principal Neighbourhood Aggregation GNN for systemic risk.

    Architecture:
        NodeEncoder (Linear -> LayerNorm -> ReLU)
        Nx PNAConv(H, H, edge_dim=D, aggregators, scalers)
           + Residual + LayerNorm + Dropout
        Classifier  (Linear -> Linear -> 2)

    Why PNA:
        Interbank networks are hub-heavy (large degree/strength heterogeneity).
        PNA uses multiple aggregators (mean, max, min, std) with degree-based
        scalers, making it robust across varying neighbourhood sizes.  It also
        natively supports continuous edge features via `edge_dim`.
    """

    def __init__(
        self,
        in_channels: int = NODE_FEAT_DIM,
        hidden: int = 32,
        out_channels: int = 2,
        edge_dim: int = EDGE_FEAT_DIM,
        dropout: float = 0.3,
        deg: torch.Tensor = None,
        aggregators: list | None = None,
        scalers: list | None = None,
        num_layers: int = 2,
        towers: int = 2,
    ):
        super().__init__()

        if aggregators is None:
            aggregators = ["mean", "max", "min", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]

        if deg is None:
            deg = torch.ones(500, dtype=torch.long)

        self.dropout = dropout
        self.num_layers = int(max(1, min(num_layers, 3)))

        # ── Node encoder ──────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # ── PNA conv blocks ──────────────────────────────────────────
        self.conv1 = PNAConv(
            in_channels=hidden,
            out_channels=hidden,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
            towers=towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        self.norm1 = nn.LayerNorm(hidden)

        if self.num_layers >= 2:
            self.conv2 = PNAConv(
                in_channels=hidden,
                out_channels=hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=edge_dim,
                towers=towers,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.norm2 = nn.LayerNorm(hidden)
        else:
            self.conv2 = None
            self.norm2 = None

        if self.num_layers >= 3:
            self.conv3 = PNAConv(
                in_channels=hidden,
                out_channels=hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=edge_dim,
                towers=towers,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.norm3 = nn.LayerNorm(hidden)
        else:
            self.conv3 = None
            self.norm3 = None

        # ── Classifier head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x:          (N, in_channels) node features
            edge_index: (2, E) COO edge indices
            edge_attr:  (E, edge_dim) edge features  [REQUIRED for PNA]
        """
        h = self.node_encoder(x)

        # Conv block 1 + residual
        h_res = h
        h = self.conv1(h, edge_index, edge_attr)
        h = self.norm1(h + h_res)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.num_layers >= 2 and self.conv2 is not None:
            # Conv block 2 + residual
            h_res = h
            h = self.conv2(h, edge_index, edge_attr)
            h = self.norm2(h + h_res)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        if self.num_layers >= 3 and self.conv3 is not None:
            # Conv block 3 + residual
            h_res = h
            h = self.conv3(h, edge_index, edge_attr)
            h = self.norm3(h + h_res)
            h = F.relu(h)

        logits = self.classifier(h)
        return logits

    def predict_proba(self, x, edge_index, edge_attr=None):
        """Return softmax probabilities (N x 2)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_attr)
            return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
#  Backward-compatible alias so `from ml_pipeline import SystRiskGCN` works
# ---------------------------------------------------------------------------
SystRiskGCN = SystRiskPNA


# ===========================================================================
#  Degree Histogram Helper
# ===========================================================================

def compute_deg_histogram(dataset: list) -> torch.Tensor:
    """Compute in-degree histogram across all graphs in the dataset.
    Required by PNAConv scalers for normalisation."""
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        for v in d:
            deg[int(v)] += 1
    return deg


def _find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
    """Find probability threshold that maximizes F1 on validation data."""
    best_th = 0.5
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    for th in np.linspace(0.05, 0.95, 19):
        preds = (probs >= th).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
            best_p = float(p)
            best_r = float(r)
    return best_th, best_p, best_r, best_f1


# ===========================================================================
#  Regression Training  (aggregated single-graph mode)
# ===========================================================================

def train_model_regression(
    agg_data: Data,
    epochs: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    patience: int = 40,
    warmup_epochs: int = 15,
    max_grad_norm: float = 1.0,
    hidden: int = 128,
    num_layers: int = 3,
    towers: int = 4,
    val_frac: float = 0.2,
    dropout: float = 0.2,
) -> SystRiskPNA:
    """
    Train PNA GNN as a *regression* model on the aggregated single graph
    using NeighborLoader mini-batching to fit large models in GPU memory.

    Key design:
      - NeighborLoader: samples k-hop subgraphs per mini-batch (avoids OOM)
      - Weighted MSE: 5× on high-risk, 2× on mid-risk nodes
      - Early stopping by Pearson r
      - Automatic threshold sweep at evaluation
    """
    from torch_geometric.loader import NeighborLoader

    print("\n" + "=" * 60)
    print("GNN REGRESSION TRAINING  (Aggregated Risk Frequency v2)")
    print("=" * 60)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    N = agg_data.num_nodes
    y = agg_data.y.float()   # (N,) continuous risk frequency

    print(f"  Nodes: {N}  |  Edges: {agg_data.edge_index.shape[1]}")
    print(f"  Target stats — mean: {y.mean():.4f}  std: {y.std():.4f}")

    # ── Node-level train/val split ────────────────────────────────────
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[perm[:n_val]] = True
    train_mask = ~val_mask

    # Store masks on the Data object for NeighborLoader
    agg_data.train_mask = train_mask
    agg_data.val_mask = val_mask

    print(f"  Train nodes: {train_mask.sum().item()}  |  Val nodes: {val_mask.sum().item()}")
    print(f"  Train target mean: {y[train_mask].mean():.4f}  |  Val target mean: {y[val_mask].mean():.4f}")

    # ── Degree histogram (single graph) ───────────────────────────────
    from torch_geometric.utils import degree as _degree
    d = _degree(agg_data.edge_index[1], num_nodes=N, dtype=torch.long)
    max_deg = int(d.max())
    deg = torch.zeros(max_deg + 1, dtype=torch.long)
    for v in d:
        deg[int(v)] += 1
    print(f"  Degree histogram: max_deg={max_deg}")

    # ── Model — single output for regression ──────────────────────────
    in_channels = agg_data.x.shape[1]
    edge_dim = agg_data.edge_attr.shape[1] if agg_data.edge_attr is not None else EDGE_FEAT_DIM

    model = SystRiskPNA(
        in_channels=in_channels,
        hidden=hidden,
        out_channels=1,
        edge_dim=edge_dim,
        deg=deg,
        num_layers=num_layers,
        towers=towers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: SystRiskPNA (regression v2)  |  Parameters: {n_params:,}")

    # ── Per-node loss weights ─────────────────────────────────────────
    node_weights = torch.ones(N)
    high_risk_mask = y >= 0.25
    node_weights[high_risk_mask] = 5.0
    mid_risk_mask = (y >= 0.15) & (y < 0.25)
    node_weights[mid_risk_mask] = 2.0
    agg_data.node_weight = node_weights   # attach to Data for NeighborLoader
    print(f"  Loss weights: 5× for {high_risk_mask.sum().item()} high-risk nodes, "
          f"2× for {mid_risk_mask.sum().item()} mid-risk, 1× for rest")

    # ── NeighborLoader for mini-batch training ────────────────────────
    # Sample neighbors per layer: fewer in deeper layers to control memory
    num_neighbors = [20, 15, 10][:num_layers]
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = val_mask.nonzero(as_tuple=False).view(-1)

    train_loader = NeighborLoader(
        agg_data,
        num_neighbors=num_neighbors,
        batch_size=512,
        input_nodes=train_idx,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        agg_data,
        num_neighbors=num_neighbors,
        batch_size=1024,
        input_nodes=val_idx,
        shuffle=False,
    )
    print(f"  NeighborLoader: neighbors={num_neighbors}, train_batch=512, val_batch=1024")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # ── MLflow ────────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ENCS-GNN-Risk")
        mlflow.start_run(run_name=f"pna_reg_v2_e{epochs}_h{hidden}_l{num_layers}")
        mlflow.log_params({
            "mode": "regression_aggregated_v2_minibatch",
            "epochs": epochs, "lr": lr, "hidden": hidden,
            "num_layers": num_layers, "towers": towers,
            "dropout": dropout, "n_nodes": N, "in_channels": in_channels,
            "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
            "n_params": n_params, "device": device,
            "num_neighbors": str(num_neighbors),
        })

    # ── Training loop ─────────────────────────────────────────────────
    best_val_r = -1.0
    best_state = None
    patience_counter = 0
    history: dict = {
        "epoch": [], "train_mse": [], "val_mse": [], "val_mae": [],
        "val_pearson_r": [], "val_f1_at_025": [], "val_auc": [], "lr": [],
    }

    for epoch in range(1, epochs + 1):
        # Warmup
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (epoch / warmup_epochs)

        # ── Train (mini-batch) ────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            ea = batch.edge_attr.float() if batch.edge_attr is not None else None
            logits = model(batch.x.float(), batch.edge_index, ea).squeeze(-1)
            preds = torch.sigmoid(logits)

            # Only compute loss on seed nodes (batch_size first nodes)
            n_seed = batch.batch_size if hasattr(batch, 'batch_size') else batch.num_nodes
            preds_seed = preds[:n_seed]
            y_seed = batch.y[:n_seed].float()
            w_seed = batch.node_weight[:n_seed].to(device)

            residuals = (preds_seed - y_seed) ** 2
            loss = (residuals * w_seed).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * n_seed
            total_nodes += n_seed

        if total_nodes == 0:
            print(f"  ⚠  Epoch {epoch}: all batches NaN — skipping.")
            continue

        train_mse = total_loss / total_nodes

        # ── Validation (mini-batch) ───────────────────────────────────
        model.eval()
        val_preds_list = []
        val_true_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ea = batch.edge_attr.float() if batch.edge_attr is not None else None
                logits = model(batch.x.float(), batch.edge_index, ea).squeeze(-1)
                preds = torch.sigmoid(logits)

                n_seed = batch.batch_size if hasattr(batch, 'batch_size') else batch.num_nodes
                val_preds_list.append(preds[:n_seed].cpu().numpy())
                val_true_list.append(batch.y[:n_seed].cpu().numpy())

        val_pred = np.concatenate(val_preds_list)
        val_true = np.concatenate(val_true_list)

        val_mse = float(np.mean((val_pred - val_true) ** 2))
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        val_r = float(np.corrcoef(val_pred, val_true)[0, 1]) if val_pred.std() > 1e-8 else 0.0

        # Binary metrics at threshold 0.25
        bin_true = (val_true >= 0.25).astype(np.int64)
        bin_pred = (val_pred >= 0.25).astype(np.int64)
        _, _, val_f1, _ = precision_recall_fscore_support(
            bin_true, bin_pred, average="binary", zero_division=0
        )
        try:
            val_auc = roc_auc_score(bin_true, val_pred)
        except ValueError:
            val_auc = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_mse"].append(round(train_mse, 6))
        history["val_mse"].append(round(val_mse, 6))
        history["val_mae"].append(round(val_mae, 6))
        history["val_pearson_r"].append(round(val_r, 4))
        history["val_f1_at_025"].append(round(float(val_f1), 4))
        history["val_auc"].append(round(float(val_auc), 4))
        history["lr"].append(round(current_lr, 8))

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "train_mse": train_mse, "val_mse": val_mse,
                "val_mae": val_mae, "val_r": val_r,
                "val_f1_025": float(val_f1), "val_auc": float(val_auc),
                "lr": current_lr,
            }, step=epoch)

        # ── Early stopping by Pearson r ───────────────────────────────
        if val_r > best_val_r + 1e-4:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no r improvement for {patience} epochs)")
                break

        if epoch > warmup_epochs:
            scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | Loss: {train_mse:.6f} | "
                f"Val MSE: {val_mse:.6f} | MAE: {val_mae:.4f} | "
                f"r: {val_r:.3f} | F1@.25: {val_f1:.3f} | AUC: {val_auc:.3f} | "
                f"LR: {current_lr:.2e}"
            )

    # ── Restore best ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    print(f"\n  Best Val Pearson r: {best_val_r:.4f}")

    # ── Final evaluation (mini-batch) ─────────────────────────────────
    model.eval()
    val_preds_list = []
    val_true_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            ea = batch.edge_attr.float() if batch.edge_attr is not None else None
            logits = model(batch.x.float(), batch.edge_index, ea).squeeze(-1)
            preds = torch.sigmoid(logits)
            n_seed = batch.batch_size if hasattr(batch, 'batch_size') else batch.num_nodes
            val_preds_list.append(preds[:n_seed].cpu().numpy())
            val_true_list.append(batch.y[:n_seed].cpu().numpy())

    val_pred_f = np.concatenate(val_preds_list)
    val_true_f = np.concatenate(val_true_list)

    final_mse = float(np.mean((val_pred_f - val_true_f) ** 2))
    final_mae = float(np.mean(np.abs(val_pred_f - val_true_f)))
    final_r = float(np.corrcoef(val_pred_f, val_true_f)[0, 1]) if val_pred_f.std() > 1e-8 else 0.0

    # ── Threshold sweep: find optimal binary threshold ────────────────
    print("\n  Threshold sweep:")
    best_th = 0.25
    best_f1_sweep = 0.0
    for th in np.arange(0.10, 0.40, 0.01):
        bt = (val_true_f >= th).astype(np.int64)
        bp = (val_pred_f >= th).astype(np.int64)
        _, _, f1_s, _ = precision_recall_fscore_support(bt, bp, average="binary", zero_division=0)
        if f1_s > best_f1_sweep:
            best_f1_sweep = f1_s
            best_th = th
    print(f"  Best threshold: {best_th:.2f} (F1={best_f1_sweep:.3f})")

    bin_true_f = (val_true_f >= best_th).astype(np.int64)
    bin_pred_f = (val_pred_f >= best_th).astype(np.int64)
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(
        bin_true_f, bin_pred_f, average="binary", zero_division=0
    )
    try:
        final_auc = roc_auc_score(bin_true_f, val_pred_f)
    except ValueError:
        final_auc = 0.0

    cm = confusion_matrix(bin_true_f, bin_pred_f)

    print(f"\n  Classification Report (Val, threshold={best_th:.2f}):")
    print(classification_report(
        bin_true_f, bin_pred_f, target_names=["Low Risk", "High Risk"], zero_division=0
    ))
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n  Final — MSE: {final_mse:.6f} | MAE: {final_mae:.4f} | "
          f"r: {final_r:.3f} | Prec: {final_prec:.3f} | Rec: {final_rec:.3f} | "
          f"F1: {final_f1:.3f} | AUC: {final_auc:.3f}")

    # ── Save artifacts ────────────────────────────────────────────────
    history_path = str(BASE_DIR / "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")

    deg_path = str(BASE_DIR / "deg_histogram.pt")
    torch.save(deg, deg_path)

    # Save normalization stats for inference
    norm_path = str(BASE_DIR / "norm_stats.pt")
    torch.save({
        "feat_mean": agg_data.feat_mean,
        "feat_std": agg_data.feat_std,
        "ea_mean": agg_data.ea_mean,
        "ea_std": agg_data.ea_std,
        "in_channels": in_channels,
    }, norm_path)
    print(f"  Normalization stats saved: {norm_path}")

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "best_val_r": best_val_r,
            "final_mse": final_mse, "final_mae": final_mae,
            "final_r": final_r, "final_f1": final_f1,
            "final_auc": final_auc, "best_threshold": best_th,
        })
        mlflow.log_artifact(history_path)
        mlflow.log_artifact(deg_path)
        mlflow.pytorch.log_model(model, "gnn_model")
        mlflow.end_run()
        print("  ✓ MLflow run logged successfully")

    return model


# ===========================================================================
#  Training  (legacy multi-graph classification — kept for reference)
# ===========================================================================

def train_model(
    dataset: list,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 1,
    device: str = "auto",
    patience: int = 15,
    warmup_epochs: int = 5,
    max_grad_norm: float = 1.0,
    hidden: int = 32,
    num_layers: int = 2,
    towers: int = 2,
    train_frac: float = 1.0,
    val_frac: float = 1.0,
    max_train_graphs: int | None = None,
    max_val_graphs: int | None = None,
    balance_mode: str = "class_weight",
    min_risky_per_graph: int = 0,
    tune_threshold: bool = True,
) -> SystRiskPNA:
    """
    Train the PNA GNN on the generated dataset.

    Key design choices:
      - Edge-aware message passing (PNA with edge_attr)
      - Degree-histogram-based scalers for hub robustness
      - Residual connections + LayerNorm
      - Stratified train/val split
      - sklearn balanced class weights
      - LR warmup + cosine annealing
      - Gradient clipping + early stopping by val F1
      - Richer metrics (F1 / Precision / Recall / AUC)
      - Full MLflow experiment tracking
    """
    print("\n" + "=" * 60)
    print("GNN TRAINING  (PNA — Edge-Aware)")
    print("=" * 60)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Feature / label validation ────────────────────────────────────
    for i, data in enumerate(dataset[:20]):
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            raise ValueError(f"Graph {i} has NaN/inf in node features!")
        if data.edge_attr is not None and (
            torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any()
        ):
            raise ValueError(f"Graph {i} has NaN/inf in edge features!")
    print("  ✓ Feature validation passed (first 20 graphs)")

    all_graph_labels = torch.cat([d.y for d in dataset]).numpy()
    n_safe_total = (all_graph_labels == 0).sum()
    n_risky_total = (all_graph_labels == 1).sum()
    print(
        f"  Dataset label distribution: Safe={n_safe_total:,}  "
        f"Risky={n_risky_total:,}  "
        f"({n_risky_total / len(all_graph_labels) * 100:.1f}% risky)"
    )
    if n_risky_total == 0:
        raise ValueError(
            "All labels are Safe (0)! Check status_to_label() and Rust output."
        )

    # ── Compute degree histogram for PNA scalers ──────────────────────
    print("  Computing degree histogram for PNA scalers...")
    deg = compute_deg_histogram(dataset)
    print(f"  Degree histogram: max_deg={len(deg)-1}, total counts={int(deg.sum())}")

    # ── Stratified train / val split ──────────────────────────────────
    graph_majority = np.array(
        [1 if (d.y == 1).sum() > (d.y == 0).sum() else 0 for d in dataset]
    )
    from sklearn.model_selection import train_test_split

    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)),
            test_size=0.2,
            stratify=graph_majority,
            random_state=42,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)), test_size=0.2, random_state=42
        )
    rng = np.random.RandomState(42)
    if train_frac < 1.0:
        k = max(1, int(len(train_idx) * train_frac))
        train_idx = rng.choice(train_idx, size=k, replace=False)
    if val_frac < 1.0:
        k = max(1, int(len(val_idx) * val_frac))
        val_idx = rng.choice(val_idx, size=k, replace=False)

    if max_train_graphs is not None:
        k = min(max_train_graphs, len(train_idx))
        train_idx = rng.choice(train_idx, size=k, replace=False)
    if max_val_graphs is not None:
        k = min(max_val_graphs, len(val_idx))
        val_idx = rng.choice(val_idx, size=k, replace=False)

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    if min_risky_per_graph > 0:
        before = len(train_data)
        train_data = [d for d in train_data if int((d.y == 1).sum()) >= min_risky_per_graph]
        after = len(train_data)
        print(f"  Train graphs after min_risky filter (>= {min_risky_per_graph}): {after}/{before}")
        if after == 0:
            raise ValueError(
                "All training graphs removed by min_risky_per_graph filter. "
                "Lower the threshold or regenerate data."
            )
    print(f"  Train graphs: {len(train_data)}, Val graphs: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ── Class weights ──────────────────────────────────────────────────
    all_train_labels = torch.cat([d.y for d in train_data]).cpu().numpy()
    cw_balanced = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=all_train_labels
    )
    if balance_mode == "class_weight":
        cw = cw_balanced
        note = "(sklearn balanced)"
    else:
        # No additional weighting if using balanced sampling or no balancing.
        cw = np.array([1.0, 1.0])
        note = "(uniform)"
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"  Class weights: Safe={cw[0]:.2f}, Risky={cw[1]:.2f}  {note}")

    in_channels = train_data[0].x.shape[1]
    edge_dim = (
        train_data[0].edge_attr.shape[1]
        if train_data[0].edge_attr is not None
        else EDGE_FEAT_DIM
    )

    model = SystRiskPNA(
        in_channels=in_channels,
        hidden=hidden,
        edge_dim=edge_dim,
        deg=deg,
        num_layers=num_layers,
        towers=towers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: SystRiskPNA  |  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── MLflow setup ──────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ENCS-GNN-Risk")
        mlflow.start_run(run_name=f"pna_e{epochs}_lr{lr}_h{hidden}")
        mlflow.log_params(
            {
                "model_type": "SystRiskPNA",
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "patience": patience,
                "warmup_epochs": warmup_epochs,
                "max_grad_norm": max_grad_norm,
                "hidden": hidden,
                "num_layers": num_layers,
                "towers": towers,
                "train_frac": train_frac,
                "val_frac": val_frac,
                "max_train_graphs": max_train_graphs,
                "max_val_graphs": max_val_graphs,
                "balance_mode": balance_mode,
                "min_risky_per_graph": min_risky_per_graph,
                "tune_threshold": tune_threshold,
                "edge_dim": edge_dim,
                "device": device,
                "in_channels": in_channels,
                "n_train_graphs": len(train_data),
                "n_val_graphs": len(val_data),
                "n_train_nodes": int(len(all_train_labels)),
                "class_weight_safe": float(cw[0]),
                "class_weight_risky": float(cw[1]),
                "n_params": n_params,
                "aggregators": "mean,max,min,std",
                "scalers": "identity,amplification,attenuation",
            }
        )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    history: dict = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_auc": [],
        "lr": [],
    }

    # AMP disabled: PNAConv's multi-aggregator internals produce NaN under fp16 autocast.
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in range(1, epochs + 1):
        # ── Warmup LR ─────────────────────────────────────────────────
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * warmup_factor

        model.train()
        total_loss = 0
        correct = 0
        total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            sel = None
            y = batch.y
            if balance_mode == "node_sample":
                # Balanced node sampling to achieve an effective ~50/50 loss signal.
                safe_idx = (y == 0).nonzero(as_tuple=False).view(-1)
                risky_idx = (y == 1).nonzero(as_tuple=False).view(-1)
                k = int(min(safe_idx.numel(), risky_idx.numel()))
                if k > 0:
                    safe_sel = safe_idx[torch.randperm(safe_idx.numel(), device=device)[:k]]
                    risky_sel = risky_idx[torch.randperm(risky_idx.numel(), device=device)[:k]]
                    sel = torch.cat([safe_sel, risky_sel], dim=0)

            # AMP greatly reduces memory for huge edge sets.
            # edge_attr is stored as fp16 on disk; PNA edge encoder needs fp32.
            ea = batch.edge_attr.float() if batch.edge_attr is not None else None
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch.x, batch.edge_index, ea)
                if sel is not None:
                    loss = criterion(logits[sel], y[sel])
                else:
                    loss = criterion(logits, y)

            # Guard: skip batch if loss is NaN/Inf (prevents model corruption)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch.num_nodes
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total_nodes += batch.num_nodes

        if total_nodes == 0:
            print(f"  ⚠  Epoch {epoch}: all batches produced NaN loss — skipping.")
            train_acc = 0.0
            avg_loss = float('inf')
        else:
            train_acc = correct / total_nodes
            avg_loss = total_loss / total_nodes

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_preds_list: list = []
        val_labels_list: list = []
        val_probs_list: list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ea = batch.edge_attr.float() if batch.edge_attr is not None else None
                logits = model(batch.x, batch.edge_index, ea)
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                val_preds_list.append(pred.cpu().numpy())
                val_labels_list.append(batch.y.cpu().numpy())
                val_probs_list.append(probs[:, 1].cpu().numpy())

        val_preds = np.concatenate(val_preds_list)
        val_labels = np.concatenate(val_labels_list)
        val_probs = np.concatenate(val_probs_list)

        val_acc = (val_preds == val_labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="binary", zero_division=0
        )
        if tune_threshold:
            best_th, best_p, best_r, best_f1 = _find_best_threshold(val_labels, val_probs)
        else:
            best_th, best_p, best_r, best_f1 = 0.5, precision, recall, f1
        try:
            auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            auc = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_loss, 6))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(float(val_acc), 4))
        history["val_f1"].append(round(float(f1), 4))
        history["val_precision"].append(round(float(precision), 4))
        history["val_recall"].append(round(float(recall), 4))
        history["val_auc"].append(round(float(auc), 4))
        history["lr"].append(round(current_lr, 8))

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                    "val_acc": float(val_acc),
                    "val_f1": float(f1),
                    "val_precision": float(precision),
                    "val_recall": float(recall),
                    "val_auc": float(auc),
                    "lr": current_lr,
                },
                step=epoch,
            )

        # ── Early stopping ────────────────────────────────────────────
        f1_for_es = best_f1 if tune_threshold else f1
        if f1_for_es > best_val_f1:
            best_val_f1 = f1_for_es
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no F1 improvement for {patience} epochs)"
                )
                break

        if epoch > warmup_epochs:
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                f"F1: {f1_for_es:.3f} | AUC: {auc:.3f} | LR: {current_lr:.2e}"
            )

    # ── Restore best model ────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best Val F1: {best_val_f1:.3f}")

    # ── Final evaluation on val set ───────────────────────────────────
    model.eval()
    final_preds_list: list = []
    final_labels_list: list = []
    final_probs_list: list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            ea = batch.edge_attr.float() if batch.edge_attr is not None else None
            logits = model(batch.x, batch.edge_index, ea)
            probs = F.softmax(logits, dim=1)
            final_preds_list.append(logits.argmax(dim=1).cpu().numpy())
            final_labels_list.append(batch.y.cpu().numpy())
            final_probs_list.append(probs[:, 1].cpu().numpy())

    all_preds = np.concatenate(final_preds_list)
    all_labels = np.concatenate(final_labels_list)
    all_probs = np.concatenate(final_probs_list)

    if tune_threshold:
        best_th, best_p, best_r, best_f1 = _find_best_threshold(all_labels, all_probs)
        tuned_preds = (all_probs >= best_th).astype(np.int64)
    else:
        best_th, best_p, best_r, best_f1 = 0.5, None, None, None
        tuned_preds = all_preds

    report_str = classification_report(
        all_labels, tuned_preds, target_names=["Safe", "Risky"], zero_division=0
    )
    print("\n  Classification Report (Validation):")
    print(report_str)

    cm = confusion_matrix(all_labels, tuned_preds)
    print(f"  Confusion Matrix:\n{cm}")

    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(
        all_labels, tuned_preds, average="binary", zero_division=0
    )
    try:
        final_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        final_auc = 0.0
    print(
        f"\n  Final Metrics — Precision: {final_prec:.3f} | Recall: {final_rec:.3f} | "
        f"F1: {final_f1:.3f} | AUC: {final_auc:.3f}"
    )

    # ── Save training history ─────────────────────────────────────────
    history_path = str(BASE_DIR / "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")

    # ── Save degree histogram for inference-time model reconstruction ─
    deg_path = str(BASE_DIR / "deg_histogram.pt")
    torch.save(deg, deg_path)
    print(f"  Degree histogram saved: {deg_path}")

    # ── MLflow: log artifacts and model ───────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(
            {
                "best_val_f1": float(best_val_f1),
                "final_precision": float(final_prec),
                "final_recall": float(final_rec),
                "final_f1": float(final_f1),
                "final_auc": float(final_auc),
            }
        )

        cm_path = str(BASE_DIR / "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(
                {"matrix": cm.tolist(), "labels": ["Safe", "Risky"]},
                f,
                indent=2,
            )
        mlflow.log_artifact(cm_path)

        report_path = str(BASE_DIR / "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(history_path)
        mlflow.log_artifact(deg_path)
        mlflow.pytorch.log_model(model, "gnn_model")

        mlflow.end_run()
        print("  ✓ MLflow run logged successfully")

    return model


# ===========================================================================
#  Explainability (XAI)
# ===========================================================================

def explain_predictions(
    model: SystRiskPNA,
    W: np.ndarray,
    df: pd.DataFrame,
    top_k_nodes: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Run GNNExplainer on the trained model to produce per-node and per-edge
    importance masks for the most-at-risk nodes.

    Returns:
        dict with:
            explanations: list of dicts per explained node
            feature_names: list of feature column names
            edge_feature_names: list of edge feature column names

    The output is also saved as xai_explanations.json and logged to MLflow.
    """
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

    print("\n" + "=" * 60)
    print("EXPLAINABILITY (GNNExplainer)")
    print("=" * 60)

    node_feats = build_node_features(W, df)
    edge_index_np = build_edge_index(W)
    edge_attr_np = build_edge_attr(W)

    x = torch.tensor(node_feats, dtype=torch.float32).to(device)
    ei = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    ea = torch.tensor(edge_attr_np, dtype=torch.float32).to(device)

    model = model.to(device)
    model.eval()

    # Get predictions first to identify top-risk nodes
    with torch.no_grad():
        logits = model(x, ei, ea)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    risk_probs = probs[:, 1]

    top_nodes = np.argsort(risk_probs)[::-1][:top_k_nodes]

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type="object",
    )

    feature_names = [
        "log_total_assets",
        "leverage_ratio",
        "log_out_strength",
        "log_in_strength",
        "interbank_ratio",
        "log_equity_capital",
        "log_deriv_notional",
    ]
    edge_feature_names = [
        "log_exposure_amount",
        "obligor_share",
        "creditor_share",
    ]

    explanations_list = []
    for rank, node_idx in enumerate(top_nodes):
        node_idx_int = int(node_idx)
        bank_name = (
            str(df.iloc[node_idx_int]["bank_name"])[:50]
            if node_idx_int < len(df)
            else "Unknown"
        )

        try:
            explanation = explainer(x, ei, edge_attr=ea, index=node_idx_int)

            # Node feature importances
            node_mask = explanation.node_mask
            if node_mask is not None:
                if node_mask.dim() == 2:
                    feat_importance = node_mask[node_idx_int].cpu().detach().numpy()
                else:
                    feat_importance = node_mask.cpu().detach().numpy()
            else:
                feat_importance = np.zeros(len(feature_names))

            # Edge importances — find top incoming counterparties
            edge_mask = explanation.edge_mask
            top_counterparties = []
            if edge_mask is not None:
                edge_imp = edge_mask.cpu().detach().numpy()
                incoming_mask = ei[1].cpu().numpy() == node_idx_int
                incoming_indices = np.where(incoming_mask)[0]
                if len(incoming_indices) > 0:
                    incoming_imp = edge_imp[incoming_indices]
                    top_edge_idx = incoming_indices[
                        np.argsort(incoming_imp)[::-1][:5]
                    ]
                    for eidx in top_edge_idx:
                        src = int(ei[0, eidx].cpu())
                        src_name = (
                            str(df.iloc[src]["bank_name"])[:40]
                            if src < len(df)
                            else "Unknown"
                        )
                        top_counterparties.append(
                            {
                                "source_idx": src,
                                "source_name": src_name,
                                "edge_importance": float(edge_imp[eidx]),
                            }
                        )

            explanations_list.append(
                {
                    "rank": rank + 1,
                    "node_idx": node_idx_int,
                    "bank_name": bank_name,
                    "risk_prob": float(risk_probs[node_idx_int]),
                    "feature_importances": {
                        name: float(imp)
                        for name, imp in zip(feature_names, feat_importance)
                    },
                    "top_counterparties": top_counterparties,
                }
            )

            if rank < 5:
                print(
                    f"\n  #{rank+1} {bank_name}  P(risk)={risk_probs[node_idx_int]:.3f}"
                )
                sorted_feats = sorted(
                    zip(feature_names, feat_importance),
                    key=lambda t: abs(t[1]),
                    reverse=True,
                )
                for fname, fimp in sorted_feats[:3]:
                    print(f"      Feature: {fname:25s}  importance={fimp:.4f}")
                for cp in top_counterparties[:3]:
                    print(
                        f"      <- {cp['source_name']:30s}  "
                        f"edge_importance={cp['edge_importance']:.4f}"
                    )

        except Exception as exc:
            print(f"  ⚠  Explanation failed for node {node_idx_int}: {exc}")
            explanations_list.append(
                {
                    "rank": rank + 1,
                    "node_idx": node_idx_int,
                    "bank_name": bank_name,
                    "risk_prob": float(risk_probs[node_idx_int]),
                    "error": str(exc),
                }
            )

    result = {
        "explanations": explanations_list,
        "feature_names": feature_names,
        "edge_feature_names": edge_feature_names,
    }

    xai_path = str(BASE_DIR / "xai_explanations.json")
    with open(xai_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  XAI explanations saved: {xai_path}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("ENCS-GNN-Risk")
            with mlflow.start_run(run_name="xai_explanations"):
                mlflow.log_artifact(xai_path)
                mlflow.log_param("top_k_nodes", top_k_nodes)
                mlflow.log_param("explained_nodes", len(explanations_list))
            print("  ✓ XAI artifacts logged to MLflow")
        except Exception:
            pass

    return result


# ===========================================================================
#  Inference
# ===========================================================================

def load_trained_model(
    model_path: str = None,
    in_channels: int = 13,           # 7 original + 6 enriched features
    device: str = "cpu",
    hidden: int = 128,
    edge_dim: int = EDGE_FEAT_DIM,
    num_layers: int = 3,
    towers: int = 4,
    out_channels: int = 1,
) -> SystRiskPNA:
    """Load a trained PNA model from disk.

    Also loads the saved degree histogram (deg_histogram.pt) so PNA scalers
    are reconstructed correctly.

    Default out_channels=1 matches the new regression mode.
    For legacy classification models, pass out_channels=2.
    """
    if model_path is None:
        model_path = str(MODEL_PATH)

    deg_path = Path(model_path).parent / "deg_histogram.pt"
    if deg_path.exists():
        deg = torch.load(str(deg_path), map_location=device, weights_only=True)
    else:
        deg = torch.ones(500, dtype=torch.long)

    model = SystRiskPNA(
        in_channels=in_channels,
        hidden=hidden,
        out_channels=out_channels,
        edge_dim=edge_dim,
        deg=deg,
        num_layers=num_layers,
        towers=towers,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict_risk(
    model: SystRiskPNA,
    W: np.ndarray,
    df: pd.DataFrame,
    device: str = "cpu",
) -> dict:
    """
    Run GNN inference on a network topology.

    *** IMPORTANT API CONTRACT (for backend/frontend integration) ***

    Args:
        model:  a SystRiskPNA instance (loaded via load_trained_model)
        W:      dense adjacency matrix (N x N) in dollars — NOT node features
        df:     bank DataFrame aligned with W (same ordering)
        device: 'cpu' or 'cuda'

    Returns:
        dict with:
            risk_probs:    np.ndarray (N,) — probability of being Risky
            risk_labels:   np.ndarray (N,) — 'High Risk' / 'Low Risk'
            risk_scores:   np.ndarray (N,) — same as risk_probs
    """
    node_feats = build_node_features(W, df)

    # Keep inference edge construction consistent with training.
    edge_index = build_edge_index(W, topk_per_row=PRUNE_TOPK_DEFAULT)
    edge_attr = build_edge_attr(W, edge_index=edge_index)

    x = torch.tensor(node_feats, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ea = torch.tensor(edge_attr, dtype=torch.float32)

    # ── Enrich features (must match aggregate_dataset) ────────────────
    from torch_geometric.utils import degree as _deg
    n_nodes = x.shape[0]
    in_deg = _deg(ei[1], num_nodes=n_nodes, dtype=torch.float).unsqueeze(1)
    out_deg = _deg(ei[0], num_nodes=n_nodes, dtype=torch.float).unsqueeze(1)
    out_str = x[:, 2:3]
    in_str = x[:, 3:4]
    net_str = out_str - in_str
    str_ratio = out_str / (out_str + in_str + 1e-6)
    log_ta = x[:, 0:1]
    log_eq = x[:, 5:6]
    asset_equity = log_ta - log_eq
    deg_ratio = out_deg / (out_deg + in_deg + 1e-6)

    x = torch.cat([
        x,
        torch.log1p(in_deg), torch.log1p(out_deg),
        net_str, str_ratio, asset_equity, deg_ratio,
    ], dim=1)

    # ── Apply z-score normalization if stats are available ─────────────
    norm_path = BASE_DIR / "norm_stats.pt"
    if norm_path.exists():
        stats = torch.load(str(norm_path), map_location="cpu", weights_only=True)
        x = (x - stats["feat_mean"].unsqueeze(0)) / stats["feat_std"].unsqueeze(0).clamp(min=1e-6)
        ea = (ea - stats["ea_mean"].unsqueeze(0)) / stats["ea_std"].unsqueeze(0).clamp(min=1e-6)

    x = x.to(device)
    ei = ei.to(device)
    ea = ea.to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x, ei, ea)
        # Regression model (out_channels=1) vs classification (out_channels=2)
        if logits.shape[-1] == 1:
            risk_probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        else:
            probs = F.softmax(logits, dim=1).cpu().numpy()
            risk_probs = probs[:, 1]
    risk_labels = np.where(risk_probs >= 0.5, "High Risk", "Low Risk")

    return {
        "risk_probs": risk_probs,
        "risk_labels": risk_labels,
        "risk_scores": risk_probs,
    }


# ===========================================================================
#  CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ENCS GNN Risk Predictor Pipeline (PNA — Edge-Aware)"
    )
    parser.add_argument(
        "--runs", type=int, default=500, help="Number of MC runs for data gen"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument(
        "--generate-only", action="store_true", help="Only generate data"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train (requires existing data)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--hidden", type=int, default=128, help="Hidden dimension for PNA layers"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of PNA layers (1-3). Lower = faster",
    )
    parser.add_argument(
        "--towers",
        type=int,
        default=4,
        help="Number of PNA towers per layer. Lower = faster",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="Fraction of training graphs to use (0-1)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=1.0,
        help="Fraction of validation graphs to use (0-1)",
    )
    parser.add_argument(
        "--max-train-graphs",
        type=int,
        default=None,
        help="Cap number of training graphs used (overrides train-frac if smaller)",
    )
    parser.add_argument(
        "--max-val-graphs",
        type=int,
        default=None,
        help="Cap number of validation graphs used (overrides val-frac if smaller)",
    )
    parser.add_argument(
        "--balance-mode",
        type=str,
        default="class_weight",
        choices=["class_weight", "node_sample", "none"],
        help="Imbalance handling: class_weight (default), node_sample, or none",
    )
    parser.add_argument(
        "--min-risky-per-graph",
        type=int,
        default=0,
        help="Filter training graphs with fewer risky nodes than this threshold",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Disable validation threshold tuning (use fixed 0.5)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if mlflow is installed",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run GNNExplainer after training and save XAI artifacts",
    )
    parser.add_argument(
        "--explain-top-k",
        type=int,
        default=10,
        help="Number of top-risk nodes to explain (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel workers (default: min(cpu_count, 12) = {min(mp.cpu_count(), 8)})",
    )
    parser.add_argument(
        "--prune-topk",
        type=int,
        default=PRUNE_TOPK_DEFAULT,
        help="Keep only top-K outgoing edges per node during training/inference (reduces GPU OOM)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=True,
        help="Aggregate MC graphs into single risk-frequency graph and train regression (recommended, default)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Disable aggregation — use legacy per-graph classification (not recommended)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the model",
    )
    args = parser.parse_args()

    if args.no_aggregate:
        args.aggregate = False

    global MLFLOW_AVAILABLE
    if args.no_mlflow:
        MLFLOW_AVAILABLE = False

    t_start = time.time()

    if not args.train_only:
        dataset = generate_dataset(
            n_runs=args.runs,
            n_workers=args.workers,
            verbose=True,
            prune_topk=args.prune_topk,
        )
        torch.save(dataset, str(DATASET_PATH))
        print(f"\n  Dataset saved: {DATASET_PATH}")
        if args.generate_only:
            return

    if args.train_only:
        if not DATASET_PATH.exists():
            print(
                f"ERROR: No dataset found at {DATASET_PATH}. "
                "Run with --generate-only first."
            )
            return
        dataset = torch.load(str(DATASET_PATH), weights_only=False)
        print(f"  Loaded dataset: {len(dataset)} graphs from {DATASET_PATH}")

    if args.aggregate:
        # ── Aggregate MC runs → single risk-frequency graph ────────
        agg_data = aggregate_dataset(dataset)
        model = train_model_regression(
            agg_data,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            hidden=args.hidden,
            num_layers=args.num_layers,
            towers=args.towers,
            device="auto",
            dropout=args.dropout,
        )
    else:
        # ── Legacy per-graph classification ────────────────────────
        model = train_model(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            hidden=args.hidden,
            num_layers=args.num_layers,
            towers=args.towers,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            max_train_graphs=args.max_train_graphs,
            max_val_graphs=args.max_val_graphs,
            balance_mode=args.balance_mode,
            min_risky_per_graph=args.min_risky_per_graph,
            tune_threshold=not args.no_threshold_tuning,
        )

    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"\n  Model saved: {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("SANITY CHECK — Inference on base network")
    print("=" * 60)

    W_sparse, df = sim.load_and_align_network()
    W_dense = sim.rescale_matrix_to_dollars(W_sparse, df)

    pred = predict_risk(model, W_dense, df)
    n_high = (pred["risk_labels"] == "High Risk").sum()
    n_low = (pred["risk_labels"] == "Low Risk").sum()
    print(f"  High Risk: {n_high}")
    print(f"  Low Risk: {n_low}")
    print(f"  Mean risk probability: {pred['risk_probs'].mean():.3f}")
    print("  Top-5 riskiest banks:")
    top5 = np.argsort(pred["risk_probs"])[::-1][:5]
    for rank, idx in enumerate(top5):
        name = df.iloc[idx]["bank_name"][:40]
        prob = pred["risk_probs"][idx]
        print(f"    {rank + 1}. {name:40s} P(risk)={prob:.3f}")

    # ── Explainability ────────────────────────────────────────────────
    if args.explain:
        explain_predictions(
            model,
            W_dense,
            df,
            top_k_nodes=args.explain_top_k,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    total_time = time.time() - t_start
    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("  DONE ✓")


if __name__ == "__main__":
    main()
