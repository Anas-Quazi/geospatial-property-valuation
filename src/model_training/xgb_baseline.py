import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 2: Spatially Aware XGBoost Baseline — Training Only
# ---------------------------------------------------------------------------
# Trains XGBoost regressor on the full master dataset using all engineered
# features (structural, spatial, temporal, OOF encodings).
# Metrics and CV evaluation are done separately in Phase 3.
# Owner : Shais013
# Input : dataset/processed/kc_master_dataset_cleaned.parquet
# Output: dataset/processed/xgb_baseline_model.json
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent.parent.parent  # src/model_training → src → root


INPUT_PATH  = BASE_DIR / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet"
OUTPUT_PATH = BASE_DIR / "models" / "xgb_baseline_model.json"

# Columns to exclude from training
# - price          : target variable
# - price_per_sqft : direct target leakage (derived from price)
# - relative_price_sqft_to_zip_median : leakage (contains price signal)
# - oof_target_enc_zip_price_sqft     : leakage (price per sqft encoding)
# - geometry       : non-numeric shapely object
# - block_id       : string identifier
# - id             : row identifier
# - fold           : CV fold assignment, not a feature

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


def train_xgb_baseline(df: pd.DataFrame) -> xgb.XGBRegressor:
    features = [c for c in df.columns if c not in IGNORE_COLS]
    X = df[features]
    y = df["price"]

    print(f"\nFeature count : {len(features)}")
    print(f"Training rows : {len(X)}")
    print(f"Target        : price (mean=${y.mean():,.0f}, std=${y.std():,.0f})")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    print("\nTraining XGBoost baseline...")
    model.fit(X, y)
    print("Training complete.")

    return model, features


def save_model(model: xgb.XGBRegressor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f"\nModel saved to: {path}")


def main():
    df            = load_data(INPUT_PATH)
    model, feats  = train_xgb_baseline(df)
    save_model(model, OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("  PHASE 2 BASELINE TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Features used : {len(feats)}")
    print(f"  Model saved   : {OUTPUT_PATH.name}")
    print("  Next          : Phase 3 — Spatial Block CV + Metrics")
    print("=" * 50)


if __name__ == "__main__":
    main()
