"""
Usage:
    python ml_pipeline.py
    python ml_pipeline.py --generate-only
    python ml_pipeline.py --train-only
    python ml_pipeline.py --runs 500
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
import ctypes as _ctypes
_itt_stub = os.environ.get(
    "ENCS_ITT_STUB_PATH",
    str(Path(__file__).parent / "libittnotify_stub.so"),
)
if os.path.isfile(_itt_stub):
    _ctypes.CDLL(_itt_stub, mode=getattr(_ctypes, "RTLD_GLOBAL", 0))
import torch  
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
from torch_geometric.nn import GCNConv, global_mean_pool
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
    """Map status strings → binary labels.  1 = Risky (Default or Distressed), 0 = Safe.
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
_SHARED: dict = {}  
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
    r = rng.random()
    if r < 0.35:
        regime = "calm"
    elif r < 0.70:
        regime = "moderate"
    else:
        regime = "stressed"
    noise = 1.0 + rng.uniform(-noise_pct, noise_pct, size=W_base.shape)
    W_noisy = np.maximum(W_base * noise, 0.0)
    np.fill_diagonal(W_noisy, 0.0)
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
    node_feats = build_node_features(W_noisy, df)
    try:
        labels = status_to_label(results["status"])
    except ValueError:
        return None
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
    patience: int = 15,
    warmup_epochs: int = 5,
    max_grad_norm: float = 1.0,
) -> SystRiskGCN:
    """
    Train the GCN on the generated dataset.
    Improvements over the original:
      - Stratified train/val split
      - sklearn balanced class weights
      - LR warmup + cosine annealing
      - Gradient clipping
      - Early stopping by val F1
      - Richer metrics (F1 / Precision / Recall / AUC)
      - Full MLflow experiment tracking
      - Training history saved to JSON
    """
    print("\n" + "=" * 60)
    print("GNN TRAINING")
    print("=" * 60)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    for i, data in enumerate(dataset[:20]):
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            raise ValueError(f"Graph {i} has NaN/inf in features!")
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
            "⚠  All labels are Safe (0)! Check status_to_label() and Rust output."
        )
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
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    print(f"  Train graphs: {len(train_data)}, Val graphs: {len(val_data)}")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    all_train_labels = torch.cat([d.y for d in train_data]).cpu().numpy()
    cw = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=all_train_labels
    )
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"  Class weights (sklearn balanced): Safe={cw[0]:.2f}, Risky={cw[1]:.2f}")
    in_channels = train_data[0].x.shape[1]
    model = SystRiskGCN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ENCS-GNN-Risk")
        mlflow.start_run(run_name=f"gcn_e{epochs}_lr{lr}")
        mlflow.log_params(
            {
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "patience": patience,
                "warmup_epochs": warmup_epochs,
                "max_grad_norm": max_grad_norm,
                "device": device,
                "in_channels": in_channels,
                "n_train_graphs": len(train_data),
                "n_val_graphs": len(val_data),
                "n_train_nodes": int(len(all_train_labels)),
                "class_weight_safe": float(cw[0]),
                "class_weight_risky": float(cw[1]),
            }
        )
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
    for epoch in range(1, epochs + 1):
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
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total_nodes += batch.num_nodes
        train_acc = correct / total_nodes
        avg_loss = total_loss / total_nodes
        model.eval()
        val_preds_list: list = []
        val_labels_list: list = []
        val_probs_list: list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
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
        if f1 > best_val_f1:
            best_val_f1 = f1
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
                f"F1: {f1:.3f} | AUC: {auc:.3f} | LR: {current_lr:.2e}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best Val F1: {best_val_f1:.3f}")
    model.eval()
    final_preds_list: list = []
    final_labels_list: list = []
    final_probs_list: list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            probs = F.softmax(logits, dim=1)
            final_preds_list.append(logits.argmax(dim=1).cpu().numpy())
            final_labels_list.append(batch.y.cpu().numpy())
            final_probs_list.append(probs[:, 1].cpu().numpy())
    all_preds = np.concatenate(final_preds_list)
    all_labels = np.concatenate(final_labels_list)
    all_probs = np.concatenate(final_probs_list)
    report_str = classification_report(
        all_labels, all_preds, target_names=["Safe", "Risky"], zero_division=0
    )
    print("\n  Classification Report (Validation):")
    print(report_str)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  Confusion Matrix:\n{cm}")
    final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        final_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        final_auc = 0.0
    print(
        f"\n  Final Metrics — Precision: {final_prec:.3f} | Recall: {final_rec:.3f} | "
        f"F1: {final_f1:.3f} | AUC: {final_auc:.3f}"
    )
    history_path = str(BASE_DIR / "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")
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
                {
                    "matrix": cm.tolist(),
                    "labels": ["Safe", "Risky"],
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(cm_path)
        report_path = str(BASE_DIR / "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(history_path)
        mlflow.pytorch.log_model(model, "gnn_model")
        mlflow.end_run()
        print("  ✓ MLflow run logged successfully")
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
        "--patience", type=int, default=15, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if mlflow is installed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel workers (default: min(cpu_count, 12) = {min(mp.cpu_count(), 8)})",
    )
    args = parser.parse_args()
    global MLFLOW_AVAILABLE
    if args.no_mlflow:
        MLFLOW_AVAILABLE = False
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
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
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