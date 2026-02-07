
## 🔄 What Changed — GCN → PNA Migration Guide

### Architecture Change
The old `SystRiskGCN` (3× GCNConv) has been replaced by **`SystRiskPNA`** (3× PNAConv with residual + LayerNorm). The alias `SystRiskGCN = SystRiskPNA` exists for backward compatibility — existing `from ml_pipeline import SystRiskGCN` still works.

### New: Edge Features
The model now uses **3-dimensional edge features** on every interbank link, built automatically by the new `build_edge_attr(W)` function:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `log1p(W[i,j])` | Exposure magnitude |
| 1 | `W[i,j] / Σ_k W[i,k]` | Share of obligor i's total outgoing debt |
| 2 | `W[i,j] / Σ_k W[k,j]` | Share of creditor j's total incoming claims |

### What Consumers Need to Know

#### `predict_risk(model, W, df, device="cpu")` — **Signature UNCHANGED**
- Still takes: model, dense adjacency matrix `W`, DataFrame `df`, device
- Still returns a dict: `{"risk_probs": ndarray, "risk_labels": ndarray, "risk_scores": ndarray}`
- Edge features are built **internally** — callers don't need to compute them
- **⚠️ The old api.py had bugs** — it was passing pre-built `X` (node features) and `edge_index` instead of `W` and `df`. Both sites are now fixed.

#### `load_trained_model(model_path, device="cpu")` — **Simplified**
- No longer requires `in_channels=7` — defaults are correct
- Automatically loads deg_histogram.pt from the same directory as the model weights

#### New Artifact: `deg_histogram.pt`
- **Must be co-located** with gnn_model.pth (same directory)
- Saved automatically during training; used by PNA scalers to normalize across varying node degrees

#### New XAI Function: `explain_predictions(model, W, df, top_k_nodes=10, device="cpu")`
- Runs GNNExplainer and returns per-node feature importances + top counterparty edge importances
- Output saved to xai_explanations.json
- Can be exposed as a new API endpoint if desired

### Frontend — No Changes Required
The REST response shape from `/api/gnn-risk` is unchanged: `{scores: [{id, name, risk_score}]}`. Same for `/api/banks` (the `gnn_risk_score` field per bank).

### CLI Changes
New flags for ml_pipeline.py:
- `--hidden 64` — PNA hidden dimension
- `--explain` — run GNNExplainer after training
- `--explain-top-k 10` — how many top-risk nodes to explain

### Files Modified
| File | Change |
|------|--------|
| ml_pipeline.py | **Rewritten** — PNA architecture, edge features, XAI, degree histogram |
| api.py | Fixed 2 broken `predict_risk` calls, cleaned imports |
| dashboard.py | Updated caption "GCN" → "PNA", cleaned imports |
| ml_pipeline_old_gcn.py | Backup of previous GCN pipeline (can be deleted) |

Made changes.