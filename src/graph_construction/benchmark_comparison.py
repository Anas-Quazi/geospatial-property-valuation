import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Week 4 — Phase 3: Benchmark Comparison (XGBoost Baseline vs SpatialGraphGAT)
# ---------------------------------------------------------------------------
# Compares the Phase 2 XGBoost baseline against the trained GNN model on the
# same spatial test blocks, and reports relative error drop.
# Owners : Shais013, Yash-Chattar
# ---------------------------------------------------------------------------

TARGET_CRS_NOTE = "Comparison run on identical spatial test folds (Spatial Block CV)"


def load_xgb_metrics(path: Path) -> dict:
    """Load saved XGBoost baseline CV results (from Phase 3, Week 2)."""
    with open(path, "r") as f:
        results = json.load(f)
    print(f"✓ Loaded XGBoost baseline metrics from: {path}")
    return results


def load_gnn_metrics(path: Path) -> dict:
    """Load saved GNN evaluation metrics (from train_gnn.py output)."""
    with open(path, "r") as f:
        results = json.load(f)
    print(f"✓ Loaded GNN metrics from: {path}")
    return results


def compute_relative_improvement(xgb_val: float, gnn_val: float) -> float:
    """Percentage drop in error going from XGBoost -> GNN. Positive = improvement."""
    if xgb_val == 0:
        return 0.0
    return ((xgb_val - gnn_val) / xgb_val) * 100


def build_comparison_table(xgb_metrics: dict, gnn_metrics: dict) -> pd.DataFrame:
    rows = []
    for metric in ["rmse", "mae", "mape"]:
        xgb_val = xgb_metrics.get(metric)
        gnn_val = gnn_metrics.get(metric)

        if xgb_val is None or gnn_val is None:
            print(f"⚠ Skipping '{metric}' — missing in one of the result files.")
            continue

        improvement = compute_relative_improvement(xgb_val, gnn_val)
        rows.append({
            "Metric": metric.upper(),
            "XGBoost Baseline": xgb_val,
            "SpatialGraphGAT": gnn_val,
            "Improvement (%)": round(improvement, 2),
        })

    return pd.DataFrame(rows)


def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent

    xgb_results_path = base_dir / "dataset" / "processed" / "cv_results.json"
    gnn_results_path = base_dir / "dataset" / "processed" / "gnn_eval_results.json"
    output_path = base_dir / "dataset" / "processed" / "benchmark_comparison.csv"

    xgb_metrics = load_xgb_metrics(xgb_results_path)
    gnn_metrics = load_gnn_metrics(gnn_results_path)

    comparison_df = build_comparison_table(xgb_metrics, gnn_metrics)

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPARISON — XGBoost vs SpatialGraphGAT")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison table saved to: {output_path}")


if __name__ == "__main__":
    main()
