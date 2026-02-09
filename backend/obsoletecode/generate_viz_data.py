"""
generate_viz_data.py — Export trimmed JSON data for the Implementation page charts.
Run once:  python generate_viz_data.py
Outputs (into ../frontend/src/data/):
  - topology_summary.json     — top 60 nodes + top 200 edges for mini network graph
  - asset_distribution.json   — histogram bins of total_assets by region
  - tier_breakdown.json       — core/periphery tier counts & aggregate stats
  - liability_heatmap.json    — top 20×20 cross-exposure matrix for heatmap
  - pipeline_stats.json       — aggregate ETL / network stats
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
BASE = Path(__file__).parent
OUTPUT = BASE / "data" / "output"
FRONTEND_DATA = BASE.parent / "frontend" / "src" / "data"
FRONTEND_DATA.mkdir(parents=True, exist_ok=True)
def load():
    df = pd.read_csv(OUTPUT / "master_nodes.csv")
    W = sparse.load_npz(OUTPUT / "adjacency_matrix.npz")
    with open(OUTPUT / "network_map.json") as f:
        net = json.load(f)
    return df, W, net
def topology_summary(net, n_nodes=60, n_edges=300):
    """Top nodes + edges for a small interactive force graph."""
    nodes = net["nodes"][:n_nodes]
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in net["edges"] if e["source"] in node_ids and e["target"] in node_ids]
    edges = sorted(edges, key=lambda e: -e["weight"])[:n_edges]
    return {"nodes": nodes, "edges": edges}
def asset_distribution(df, bins=20):
    """Histogram of log10(total_assets) by region."""
    result = {}
    for region in ["US", "EU"]:
        vals = df.loc[df["region"] == region, "total_assets"]
        vals = vals[vals > 0]
        log_vals = np.log10(vals)
        counts, edges = np.histogram(log_vals, bins=bins)
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        result[region] = {
            "counts": counts.tolist(),
            "centers": [round(c, 2) for c in centers],
            "labels": [f"$10^{{{c:.1f}}}" for c in centers],
            "total_banks": int(len(vals)),
            "total_assets_T": round(float(vals.sum() / 1e12), 2),
        }
    return result
def tier_breakdown(df):
    """Compute tier-level aggregate stats."""
    EU_IB = 0.15
    US_IB = 0.10
    DERIV_MULT = 1.5
    df = df.copy()
    df["interbank_ratio"] = np.where(df["region"] == "EU", EU_IB, US_IB)
    df["interbank_assets"] = df["total_assets"] * df["interbank_ratio"]
    if "deriv_ir_notional" in df.columns:
        has_d = df["deriv_ir_notional"] > 0
        df.loc[has_d, "interbank_assets"] *= DERIV_MULT
    df = df.sort_values("interbank_assets", ascending=False).reset_index(drop=True)
    n = len(df)
    df["tier"] = "periphery"
    sc = min(30, n)
    core_end = sc + int((n - sc) * 0.05)
    df.loc[:sc - 1, "tier"] = "super_core"
    df.loc[sc:core_end - 1, "tier"] = "core"
    tiers = []
    for tier in ["super_core", "core", "periphery"]:
        sub = df[df["tier"] == tier]
        tiers.append({
            "tier": tier,
            "count": int(len(sub)),
            "total_assets_T": round(float(sub["total_assets"].sum() / 1e12), 2),
            "avg_assets_B": round(float(sub["total_assets"].mean() / 1e9), 2),
            "total_equity_T": round(float(sub["equity_capital"].sum() / 1e12), 2),
            "us_count": int((sub["region"] == "US").sum()),
            "eu_count": int((sub["region"] == "EU").sum()),
        })
    return tiers
def liability_heatmap(df, W, top_n=20):
    """Top 20×20 interbank liability matrix for a heatmap."""
    EU_IB, US_IB, DM = 0.15, 0.10, 1.5
    df = df.copy()
    df["ib_ratio"] = np.where(df["region"] == "EU", EU_IB, US_IB)
    df["ib_assets"] = df["total_assets"] * df["ib_ratio"]
    if "deriv_ir_notional" in df.columns:
        has_d = df["deriv_ir_notional"] > 0
        df.loc[has_d, "ib_assets"] *= DM
    df = df.sort_values("ib_assets", ascending=False).reset_index(drop=True)
    W_dense = W.toarray()
    n = min(top_n, W_dense.shape[0])
    sub = W_dense[:n, :n]
    ib_liab = df["total_liabilities"].values[:n] * df["ib_ratio"].values[:n]
    row_sums = sub.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 0:
            sub[i, :] *= ib_liab[i] / row_sums[i]
    log_sub = np.log10(np.maximum(sub, 1)).tolist()
    names = [str(df.iloc[i]["bank_name"])[:25] for i in range(n)]
    regions = df["region"].values[:n].tolist()
    return {
        "matrix": [[round(v, 2) for v in row] for row in log_sub],
        "names": names,
        "regions": regions,
        "raw_billions": [[round(sub[i][j] / 1e9, 2) for j in range(n)] for i in range(n)],
    }
def pipeline_stats(df, W):
    """Summary statistics for the pipeline."""
    us = df[df["region"] == "US"]
    eu = df[df["region"] == "EU"]
    return {
        "total_banks": int(len(df)),
        "us_banks": int(len(us)),
        "eu_banks": int(len(eu)),
        "total_assets_T": round(float(df["total_assets"].sum() / 1e12), 2),
        "us_assets_T": round(float(us["total_assets"].sum() / 1e12), 2),
        "eu_assets_T": round(float(eu["total_assets"].sum() / 1e12), 2),
        "total_equity_T": round(float(df["equity_capital"].sum() / 1e12), 2),
        "matrix_shape": list(W.shape),
        "matrix_nnz": int(W.nnz),
        "matrix_density_pct": round(W.nnz / (W.shape[0] * W.shape[1]) * 100, 4),
        "columns": list(df.columns),
        "n_columns": int(len(df.columns)),
        "has_derivatives": bool("deriv_ir_notional" in df.columns),
        "banks_with_derivatives": int((df.get("deriv_ir_notional", pd.Series(dtype=float)).fillna(0) > 0).sum()),
    }
def main():
    df, W, net = load()
    ts = topology_summary(net)
    with open(FRONTEND_DATA / "topology_summary.json", "w") as f:
        json.dump(ts, f)
    ad = asset_distribution(df)
    with open(FRONTEND_DATA / "asset_distribution.json", "w") as f:
        json.dump(ad, f)
    tb = tier_breakdown(df)
    with open(FRONTEND_DATA / "tier_breakdown.json", "w") as f:
        json.dump(tb, f)
    lh = liability_heatmap(df, W)
    with open(FRONTEND_DATA / "liability_heatmap.json", "w") as f:
        json.dump(lh, f)
    ps = pipeline_stats(df, W)
    with open(FRONTEND_DATA / "pipeline_stats.json", "w") as f:
        json.dump(ps, f, indent=2)
    print(f"\n written to {FRONTEND_DATA}")
if __name__ == "__main__":
    main()