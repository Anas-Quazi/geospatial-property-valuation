import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from xgb_baseline import load_data, IGNORE_COLS, INPUT_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "cv_results.json"

MODEL_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def run_spatial_cv(df: pd.DataFrame) -> dict:
    features = [c for c in df.columns if c not in IGNORE_COLS]
    folds = sorted(df["fold"].unique())

    print(f"Feature count : {len(features)}")
    print(f"Folds found   : {[int(f) for f in folds]}")
    print(f"Target        : price (mean=${df['price'].mean():,.0f}, "
          f"std=${df['price'].std():,.0f})\n")

    fold_results = []

    for k in folds:
        k_int = int(k)
        train_df = df[df["fold"] != k]
        val_df = df[df["fold"] == k]

        X_train, y_train = train_df[features], train_df["price"]
        X_val, y_val = val_df[features], val_df["price"]

        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        fold_rmse = rmse(y_val.values, preds)
        fold_mape = mape(y_val.values, preds)

        print(f"Fold {k_int} - RMSE: ${fold_rmse:,.0f} | MAPE: {fold_mape:.2f}%")

        fold_results.append({
            "fold": k_int,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "rmse": fold_rmse,
            "mape": fold_mape,
        })

    mean_rmse = float(np.mean([r["rmse"] for r in fold_results]))
    mean_mape = float(np.mean([r["mape"] for r in fold_results]))

    print(f"\nMean CV RMSE: ${mean_rmse:,.0f}")
    print(f"Mean CV MAPE: {mean_mape:.2f}%")

    return {
        "model_params": MODEL_PARAMS,
        "features_used": len(features),
        "fold_results": fold_results,
        "mean_cv_rmse": mean_rmse,
        "mean_cv_mape": mean_mape,
    }


def save_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {path}")


def main():
    df = load_data(INPUT_PATH)
    results = run_spatial_cv(df)
    save_results(results, OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("  PHASE 3 SPATIAL BLOCK CV COMPLETE")
    print("=" * 50)
    print(f"  Mean CV RMSE : ${results['mean_cv_rmse']:,.0f}")
    print(f"  Mean CV MAPE : {results['mean_cv_mape']:.2f}%")
    print(f"  Next         : Week 3 - Spatial embeddings / graph construction")
    print("=" * 50)


if __name__ == "__main__":
    main()
