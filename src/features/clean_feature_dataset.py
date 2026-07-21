import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "dataset" / "processed" / "kc_master_dataset.parquet"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet"


def clean_master_dataset(df):
    """Applies all cleaning rules to the master dataset and returns the cleaned df."""
    df = df.copy()

    # 1. Clip sqft_non_living to 0
    df["sqft_non_living"] = df["sqft_non_living"].clip(lower=0)

    # 2. Fix pre-construction ages
    df["house_age"] = df["house_age"].clip(lower=0)
    df["age_at_sale"] = df["age_at_sale"].clip(lower=0)

    # 3. Fix renovation timing
    df["years_since_renovated"] = df["years_since_renovated"].clip(lower=0)

    # 4. Cap extreme lot sizes at 99.5th percentile
    lot_cap = df["sqft_lot"].quantile(0.995)
    df["sqft_lot"] = df["sqft_lot"].clip(upper=lot_cap)

    # 5. Cap extreme price/sqft at 99.5th percentile
    price_sqft_cap = df["price_per_sqft"].quantile(0.995)
    df["price_per_sqft"] = df["price_per_sqft"].clip(upper=price_sqft_cap)

    # 6. Clip negative relative age ratio caused by pre-construction age adjustment
    df["relative_age_to_zip_median"] = df["relative_age_to_zip_median"].clip(lower=0)

    return df


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {INPUT_PATH.name}: {df.shape}")

    cleaned_df = clean_master_dataset(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_PATH} : {cleaned_df.shape}")


if __name__ == "__main__":
    main()
