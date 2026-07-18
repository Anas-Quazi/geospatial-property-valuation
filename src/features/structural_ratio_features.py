import os
import numpy as np
import pandas as pd
from pathlib import Path

REQUIRED_COLS = [
    "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "floors",
    "grade", "condition", "sqft_basement", "sqft_above",
    "sqft_living15", "sqft_lot15",
]


def add_structural_ratio_features(input_path: str, output_path: str):
    """
    Category 3: Structural Ratios & Density Metrics (14 features).

    Reads the cleaned/spatial dataset, adds the 14 Category 3 features as
    pure, per-row deterministic transforms of existing structural columns,
    and writes the result to a new parquet file. Scope is strictly
    Category 3 — no fold logic, no other category's features touched.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file missing at: {input_path}")

    print("Reading input file...")
    if str(input_path).endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    # Guard against silent div-by-zero: denominators that are 0 would
    # produce inf rather than erroring. Fail loud instead.
    zero_checks = {
        "sqft_lot": df["sqft_lot"],
        "bedrooms + bathrooms": df["bedrooms"] + df["bathrooms"],
        "bathrooms": df["bathrooms"],
        "floors": df["floors"],
        "sqft_living": df["sqft_living"],
        "sqft_living15": df["sqft_living15"],
        "sqft_lot15": df["sqft_lot15"],
        "condition": df["condition"],
    }
    for name, series in zero_checks.items():
        n_zero = int((series == 0).sum())
        if n_zero > 0:
            raise ValueError(
                f"Found {n_zero} zero value(s) in denominator '{name}'. "
                "Fix upstream data before computing ratio features."
            )

    print("Computing structural ratio & density features...")

    # 21. land_to_structure_ratio
    df["land_to_structure_ratio"] = df["sqft_living"] / df["sqft_lot"]

    # 22. sqft_non_living
    df["sqft_non_living"] = df["sqft_lot"] - df["sqft_living"]

    # 23. avg_room_size
    df["avg_room_size"] = df["sqft_living"] / (df["bedrooms"] + df["bathrooms"])

    # 24. bed_bath_ratio
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"]

    # 25. sqft_living_per_floor
    df["sqft_living_per_floor"] = df["sqft_living"] / df["floors"]

    # 26. is_mansion
    df["is_mansion"] = (
        (df["sqft_living"] > 4000) & (df["grade"] >= 10)
    ).astype(int)

    # 27. luxury_score
    df["luxury_score"] = df["grade"] * df["condition"]

    # 28. has_basement
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # 29. basement_to_living_ratio
    df["basement_to_living_ratio"] = df["sqft_basement"] / df["sqft_living"]

    # 30. above_to_living_ratio
    df["above_to_living_ratio"] = df["sqft_above"] / df["sqft_living"]

    # 31. living_to_lot15_ratio
    df["living_to_lot15_ratio"] = df["sqft_living"] / df["sqft_living15"]

    # 32. lot_to_lot15_ratio
    df["lot_to_lot15_ratio"] = df["sqft_lot"] / df["sqft_lot15"]

    # 33. total_rooms
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    # 34. grade_to_condition_ratio
    df["grade_to_condition_ratio"] = df["grade"] / df["condition"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving Category 3 feature output to: {output_path}")
    df.to_parquet(output_path, index=False)

    added_cols = [
        "land_to_structure_ratio", "sqft_non_living", "avg_room_size",
        "bed_bath_ratio", "sqft_living_per_floor", "is_mansion",
        "luxury_score", "has_basement", "basement_to_living_ratio",
        "above_to_living_ratio", "living_to_lot15_ratio",
        "lot_to_lot15_ratio", "total_rooms", "grade_to_condition_ratio",
    ]
    print(f"Category 3 complete! Added {len(added_cols)} features:")
    for c in added_cols:
        print(f"    ├─ {c}")

    return df


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent

    input_data = BASE_DIR / "dataset" / "kc_house_cleaned.csv"
    output_features = BASE_DIR / "dataset" / "processed" / "structural_ratio_features.parquet"

    add_structural_ratio_features(
        input_path=str(input_data),
        output_path=str(output_features),
    )
