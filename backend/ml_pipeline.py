"""

Usage:
    python ml_pipeline.py

    python ml_pipeline.py --generate-only

    python ml_pipeline.py --train-only

    python ml_pipeline.py --runs 500

"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import simulation_engine as sim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "output"
MODEL_PATH = BASE_DIR / "gnn_model.pth"
DATASET_PATH = DATA_DIR / "gnn_dataset.pt"


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
    n = W.shape[0]
    out_strength = W.sum(axis=1)

    in_strength = W.sum(axis=0)

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


def build_edge_index(W: np.ndarray) -> np.ndarray:
    """Convert dense adjacency to COO edge_index (2 × E)."""
    rows, cols = np.nonzero(W)
    return np.stack([rows, cols], axis=0).astype(np.int64)


def status_to_label(status_array: np.ndarray) -> np.ndarray:
    """Map status strings → binary labels.  1 = Risky (Default or Distressed), 0 = Safe."""
    labels = np.zeros(len(status_array), dtype=np.int64)
    labels[status_array == "Default"] = 1
    labels[status_array == "Distressed"] = 1
    return labels


# ---------------------------------------------------------------------------
# Multiprocessing helpers — data shared via fork (copy-on-write on Linux)
# ---------------------------------------------------------------------------

_SHARED: dict = {}  # populated by generate_dataset before pool creation


def _mp_worker_init():
    """Limit per-worker OpenMP / MKL threads to avoid contention."""
    torch.set_num_threads(1)


def _single_mc_run(args):
    """
    Execute one Monte Carlo simulation run.
    Designed for multiprocessing.Pool — reads from module-level _SHARED dict
    which is inherited via fork.
    """
    run_idx, noise_pct, n_steps, seed_base = args

    W_base = _SHARED["W_base"]
    df = _SHARED["df"]
    edge_index_base = _SHARED["edge_index"]
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

    # ── Perturb adjacency matrix ──────────────────────────────────────
    noise = 1.0 + rng.uniform(-noise_pct, noise_pct, size=W_base.shape)
    W_noisy = np.maximum(W_base * noise, 0.0)
    np.fill_diagonal(W_noisy, 0.0)

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
    except Exception:
        return None

    # ── Build PyG graph ───────────────────────────────────────────────
    node_feats = build_node_features(W_noisy, df)
    labels = status_to_label(results["status"])

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_base, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
    )
    return (data, regime)


def generate_dataset(
    n_runs: int = 500,
    noise_pct: float = 0.10,
    n_steps: int = 5,
    n_workers: int | None = None,
    verbose: bool = True,
) -> list:
    """
    Run the simulation `n_runs` times with random perturbations.
    Uses multiprocessing to parallelise across CPU cores.
    Returns a list of PyG Data objects.

    Runs are split into three regimes so the model sees genuine Safe labels:
      - CALM   (35%): tiny severity, no margin spirals, low fire-sale α,
                       small-bank triggers → most nodes survive.
      - MODERATE (35%): medium severity, mild margins & fire-sales,
                        mixed trigger pool → partial cascade.
      - STRESSED (30%): high severity, full margins & fire-sales,
                        top-connected triggers → mass default.
    """
    global _SHARED

    print("\n" + "=" * 60)
    print(f"GNN DATA GENERATION — {n_runs} Monte Carlo runs  (3 regimes)")
    print("=" * 60)

    # ── Load data once in the main process ────────────────────────────
    W_sparse, df = sim.load_and_align_network()
    W_base = sim.rescale_matrix_to_dollars(W_sparse, df)
    n = W_base.shape[0]

    edge_index_base = build_edge_index(W_base)

    out_strength = W_base.sum(axis=1)
    rank = np.argsort(out_strength)[::-1]
    pool_top30 = rank[: min(30, n)]
    pool_mid = rank[min(30, n) : min(200, n)]
    pool_small = rank[min(200, n) :]
    pool_all = np.arange(n)

    # ── Shared data — inherited by workers via fork (COW) ─────────────
    _SHARED.update(
        {
            "W_base": W_base,
            "df": df,
            "edge_index": edge_index_base,
            "pool_top30": pool_top30,
            "pool_mid": pool_mid,
            "pool_small": pool_small,
            "pool_all": pool_all,
        }
    )

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 12)
    # Guarantee at least 1 worker
    n_workers = max(1, n_workers)

    args_list = [(i, noise_pct, n_steps, 42) for i in range(n_runs)]
    chunksize = max(1, n_runs // (n_workers * 8))

    dataset: list = []
    label_counts = {0: 0, 1: 0}
    regime_counts = {"calm": 0, "moderate": 0, "stressed": 0}
    failed = 0
    t0 = time.time()

    print(f"  Workers: {n_workers}  |  Chunksize: {chunksize}")

    with mp.Pool(processes=n_workers, initializer=_mp_worker_init) as pool:
        results_iter = pool.imap_unordered(
            _single_mc_run, args_list, chunksize=chunksize
        )
        for i, result in enumerate(results_iter):
            if result is not None:
                data, regime = result
                dataset.append(data)
                regime_counts[regime] += 1
                for lbl in data.y.numpy():
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
        print(f"  Class balance: {label_counts[1] / total * 100:.1f}% risky")

    return dataset


class SystRiskGCN(nn.Module):
    """
    3-layer Graph Convolutional Network for node-level risk classification.
    Architecture: GCNConv(in→64) → GCNConv(64→32) → GCNConv(32→16) → Linear(16→2)
    """

    def __init__(
        self,
        in_channels: int,
        hidden1: int = 64,
        hidden2: int = 32,
        hidden3: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, hidden3)
        self.classifier = nn.Linear(hidden3, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        logits = self.classifier(x)
        return logits

    def predict_proba(self, x, edge_index):
        """Return softmax probabilities (N × 2)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=1)


def train_model(
    dataset: list,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    device: str = "auto",
) -> SystRiskGCN:
    """
    Train the GCN on the generated dataset.
    Uses class-weighted cross-entropy to handle label imbalance.
    """
    print("\n" + "=" * 60)
    print("GNN TRAINING")
    print("=" * 60)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    val_data = dataset[n_train:]
    print(f"  Train graphs: {len(train_data)}, Val graphs: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    all_labels = torch.cat([d.y for d in train_data])
    n_safe = (all_labels == 0).sum().float()
    n_risky = (all_labels == 1).sum().float()
    total = n_safe + n_risky

    w_safe = min(total / (2 * n_safe + 1), 5.0)
    w_risky = min(total / (2 * n_risky + 1), 5.0)
    class_weights = torch.tensor([w_safe, w_risky], dtype=torch.float32).to(device)
    print(f"  Class weights: Safe={w_safe:.2f}, Risky={w_risky:.2f}")

    in_channels = train_data[0].x.shape[1]
    model = SystRiskGCN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total_nodes += batch.num_nodes

        train_acc = correct / total_nodes
        avg_loss = total_loss / total_nodes

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.num_nodes

        val_acc = val_correct / val_total if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best Val Accuracy: {best_val_acc:.3f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print("\n  Classification Report (Validation):")
    print(
        classification_report(
            all_labels, all_preds, target_names=["Safe", "Risky"], zero_division=0
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    print(f"  Confusion Matrix:\n{cm}")

    return model


def load_trained_model(
    model_path: str = None, in_channels: int = 7, device: str = "cpu"
) -> SystRiskGCN:
    """Load a trained GCN from disk."""
    if model_path is None:
        model_path = str(MODEL_PATH)
    model = SystRiskGCN(in_channels=in_channels)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict_risk(
    model: SystRiskGCN, W: np.ndarray, df: pd.DataFrame, device: str = "cpu"
) -> dict:
    """
    Run GNN inference on a network topology.

    Returns:
        dict with:
            risk_probs:    np.ndarray (N,) — probability of being Risky
            risk_labels:   np.ndarray (N,) — 'High Risk' / 'Low Risk'
            risk_scores:   np.ndarray (N,) — same as risk_probs (for coloring)
    """
    node_feats = build_node_features(W, df)
    edge_index = build_edge_index(W)

    x = torch.tensor(node_feats, dtype=torch.float32).to(device)
    ei = torch.tensor(edge_index, dtype=torch.long).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x, ei)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    risk_probs = probs[:, 1]

    risk_labels = np.where(risk_probs >= 0.5, "High Risk", "Low Risk")

    return {
        "risk_probs": risk_probs,
        "risk_labels": risk_labels,
        "risk_scores": risk_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="ENCS GNN Risk Predictor Pipeline")
    parser.add_argument(
        "--runs", type=int, default=500, help="Number of MC runs for data gen"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument(
        "--generate-only", action="store_true", help="Only generate data"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only train (requires existing data)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel workers (default: min(cpu_count, 12) = {min(mp.cpu_count(), 8)})",
    )
    args = parser.parse_args()

    t_start = time.time()

    if not args.train_only:
        dataset = generate_dataset(
            n_runs=args.runs, n_workers=args.workers, verbose=True
        )

        torch.save(dataset, str(DATASET_PATH))
        print(f"\n  Dataset saved: {DATASET_PATH}")
        if args.generate_only:
            return

    if args.train_only:
        if not DATASET_PATH.exists():
            print(
                f"ERROR: No dataset found at {DATASET_PATH}. Run with --generate-only first."
            )
            return
        dataset = torch.load(str(DATASET_PATH), weights_only=False)
        print(f"  Loaded dataset: {len(dataset)} graphs from {DATASET_PATH}")

    model = train_model(
        dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
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
    print(f"  Top-5 riskiest banks:")
    top5 = np.argsort(pred["risk_probs"])[::-1][:5]
    for rank, idx in enumerate(top5):
        name = df.iloc[idx]["bank_name"][:40]
        prob = pred["risk_probs"][idx]
        print(f"    {rank + 1}. {name:40s} P(risk)={prob:.3f}")

    total_time = time.time() - t_start
    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("  DONE ✓")


if __name__ == "__main__":
    main()
