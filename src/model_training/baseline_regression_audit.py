import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Phase 3: Comprehensive Baseline Regression Audit
# ---------------------------------------------------------------------------
# Evaluates full regression performance metrics across spatial folds:
# - RMSE, MAE, MedAE, MAPE, R2, and Max Error
# Owner : Shais013
# Input : dataset/processed/kc_master_dataset_cleaned.parquet
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent  # Go up to project root

INPUT_PATH = BASE_DIR / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet"
OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "full_regression_audit.json"

IGNORE_COLS = [
    "price",
    "price_per_sqft",
    "relative_price_sqft_to_zip_median",
    "oof_target_enc_zip_price_sqft",
    "oof_target_enc_zip_price",
    "geometry",
    "block_id",
    "id",
    "fold",
]


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")
    return df


def run_full_audit(df: pd.DataFrame):
    features = [c for c in df.columns if c not in IGNORE_COLS]
    X = df[features]
    y = df["price"]
    folds = sorted(df["fold"].unique())

    print(f"\nRunning full regression audit across {len(folds)} spatial folds...")

    metrics_per_fold = []

    for fold_id in folds:
        train_mask = df["fold"] != fold_id
        val_mask = df["fold"] == fold_id

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Calculate full suite of regression metrics
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        medae = median_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds) * 100
        r2 = r2_score(y_val, preds) * 100
        max_err = np.max(np.abs(y_val - preds))
        p90_err = np.percentile(np.abs(y_val - preds), 90)

        fold_stats = {
            "fold": int(fold_id),
            "rmse": float(rmse),
            "mae": float(mae),
            "medae": float(medae),
            "mape": float(mape),
            "r2_score_pct": float(r2),
            "p90_error": float(p90_err),
            "max_error": float(max_err),
        }
        metrics_per_fold.append(fold_stats)

        print(
            f"Fold {fold_id} -> R²: {r2:.2f}% | MAE: ${mae:,.0f} | MedAE: ${medae:,.0f} | RMSE: ${rmse:,.0f} | MAPE: {mape:.2f}%"
        )

    # Compute overall mean metrics
    avg_r2 = np.mean([f["r2_score_pct"] for f in metrics_per_fold])
    avg_mae = np.mean([f["mae"] for f in metrics_per_fold])
    avg_medae = np.mean([f["medae"] for f in metrics_per_fold])
    avg_rmse = np.mean([f["rmse"] for f in metrics_per_fold])
    avg_mape = np.mean([f["mape"] for f in metrics_per_fold])

    print("\n" + "=" * 55)
    print("         FINAL BASELINE REGRESSION AUDIT")
    print("=" * 55)
    print(f"  R² Score (Variance Explained) : {avg_r2:.2f}%")
    print(f"  Mean Absolute Error (MAE)     : ${avg_mae:,.2f}")
    print(f"  Median Absolute Error (MedAE) : ${avg_medae:,.2f}")
    print(f"  Root Mean Squared Error (RMSE): ${avg_rmse:,.2f}")
    print(f"  Mean Absolute % Error (MAPE)  : {avg_mape:.2f}%")
    print("=" * 55)

    # Save summary audit results to JSON
    audit_summary = {
        "mean_r2_pct": avg_r2,
        "mean_mae": avg_mae,
        "mean_medae": avg_medae,
        "mean_rmse": avg_rmse,
        "mean_mape": avg_mape,
        "fold_details": metrics_per_fold,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(audit_summary, f, indent=4)
    print(f"\nAudit saved to: {OUTPUT_PATH}")


def main():
    df = load_data(INPUT_PATH)
    run_full_audit(df)


if __name__ == "__main__":
    main()