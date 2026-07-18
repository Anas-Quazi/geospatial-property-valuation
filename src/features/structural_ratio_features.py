import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path


# ---------------------------------------------------------------------------
# Category 3: Structural Ratios & Internal Composition (Features 21–34)
# ---------------------------------------------------------------------------
# These features capture how a house is internally composed — the relationship
# between its physical components (rooms, floors, area, grade, condition).
# They are pure per-row deterministic transforms derived exclusively from
# structural columns already in the dataset. No spatial context, no fold
# logic, no target encoding — zero leakage risk.
#
# Input : spatial_proximity_features.parquet  (output of Category 2)
# Output: structural_ratio_features.parquet   (Category 3 appended)
# Owner : Shais013
# ---------------------------------------------------------------------------

# CRS constants — must stay consistent across all feature modules
PROJECTED_CRS = "EPSG:2285"


def add_structural_ratio_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Category 3: Structural Ratios & Internal Composition (14 features, #21–34).

    All features are safe, deterministic, per-row transforms. No external
    reference points, no fold-level statistics, no target values used.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain at minimum: bedrooms, bathrooms, sqft_living, sqft_lot,
        sqft_above, sqft_basement, floors, grade, condition,
        sqft_living15, sqft_lot15.

    Returns
    -------
    GeoDataFrame with 14 new columns appended.
    """
    gdf = gdf.copy()

    required_cols = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "sqft_above", "sqft_basement", "floors", "grade",
        "condition", "sqft_living15", "sqft_lot15",
    ]
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for Category 3: {missing}. "
            "Ensure the input parquet was produced by the Category 2 pipeline."
        )

    # Guard against null propagation in structural columns
    null_counts = gdf[required_cols].isnull().sum()
    bad_cols = null_counts[null_counts > 0]
    if not bad_cols.empty:
        raise ValueError(
            f"Null values found in structural columns: {bad_cols.to_dict()}. "
            "Fix upstream before computing ratios."
        )

    # ------------------------------------------------------------------
    # Feature 21: bath_per_bed
    # Bathroom-to-bedroom ratio. +1 guard avoids division by zero for
    # studio/0-bedroom listings.
    # ------------------------------------------------------------------
    gdf["bath_per_bed"] = gdf["bathrooms"] / (gdf["bedrooms"] + 1)

    # ------------------------------------------------------------------
    # Feature 22: sqft_per_bed
    # Average living area allocated per bedroom. Higher = more spacious.
    # +1 guard for 0-bedroom listings (studios).
    # ------------------------------------------------------------------
    gdf["sqft_per_bed"] = gdf["sqft_living"] / (gdf["bedrooms"] + 1)

    # ------------------------------------------------------------------
    # Feature 23: sqft_per_bath
    # Average living area per bathroom. Captures bathroom density relative
    # to total space — relevant for luxury vs budget segmentation.
    # +1 guard for 0-bathroom edge cases.
    # ------------------------------------------------------------------
    gdf["sqft_per_bath"] = gdf["sqft_living"] / (gdf["bathrooms"] + 1)

    # ------------------------------------------------------------------
    # Feature 24: above_to_living_ratio
    # Fraction of total living area that is above ground.
    # = 1.0 for houses with no basement; < 1.0 for partial/full basements.
    # sqft_living is always > 0 (enforced by preprocessing outlier removal).
    # ------------------------------------------------------------------
    gdf["above_to_living_ratio"] = gdf["sqft_above"] / gdf["sqft_living"]

    # ------------------------------------------------------------------
    # Feature 25: basement_to_living_ratio
    # Fraction of total living area below ground.
    # = 0.0 for houses with no basement.
    # ------------------------------------------------------------------
    gdf["basement_to_living_ratio"] = gdf["sqft_basement"] / gdf["sqft_living"]

    # ------------------------------------------------------------------
    # Feature 26: lot_to_living_ratio
    # How much land exists relative to the built footprint.
    # High values = large lot relative to house (suburban/rural).
    # Low values = dense urban or large house on small plot.
    # ------------------------------------------------------------------
    gdf["lot_to_living_ratio"] = gdf["sqft_lot"] / gdf["sqft_living"]

    # ------------------------------------------------------------------
    # Feature 27: living_to_lot_pct
    # Percentage of the lot that is occupied by the house footprint.
    # Inverse perspective of Feature 26 — emphasises how built-up the lot is.
    # Capped implicitly by lot_to_living_ratio being >= 1 in most cases.
    # ------------------------------------------------------------------
    gdf["living_to_lot_pct"] = (gdf["sqft_living"] / gdf["sqft_lot"]) * 100

    # ------------------------------------------------------------------
    # Feature 28: floor_area_efficiency
    # Average square footage per floor. Multi-storey houses with small
    # floor plates have lower values; single-storey sprawling homes are high.
    # +1 guard is not needed (floors >= 1 always), but kept for robustness.
    # ------------------------------------------------------------------
    gdf["floor_area_efficiency"] = gdf["sqft_living"] / gdf["floors"]

    # ------------------------------------------------------------------
    # Feature 29: neighbor_living_ratio
    # Subject house sqft_living relative to the average of its 15 nearest
    # neighbors (sqft_living15). Captures relative size within the micro-
    # neighborhood — a key signal for price premiums on oversized homes.
    # +1 guard for any edge case where sqft_living15 = 0.
    # ------------------------------------------------------------------
    gdf["neighbor_living_ratio"] = gdf["sqft_living"] / (gdf["sqft_living15"] + 1)

    # ------------------------------------------------------------------
    # Feature 30: neighbor_lot_ratio
    # Subject lot size relative to neighbors' average lot size (sqft_lot15).
    # Flags plots significantly larger or smaller than their surroundings.
    # +1 guard for edge cases.
    # ------------------------------------------------------------------
    gdf["neighbor_lot_ratio"] = gdf["sqft_lot"] / (gdf["sqft_lot15"] + 1)

    # ------------------------------------------------------------------
    # Feature 31: grade_condition_score
    # Multiplicative interaction of grade (build quality) and condition
    # (maintenance state). A high-grade, well-maintained house scores
    # disproportionately higher than either component alone.
    # Both are ordinal ints from the source data.
    # ------------------------------------------------------------------
    gdf["grade_condition_score"] = gdf["grade"] * gdf["condition"]

    # ------------------------------------------------------------------
    # Feature 32: grade_per_sqft
    # Build quality index normalized by total living area.
    # High values = high-grade small home (boutique/luxury).
    # Low values = moderate-grade large home (functional/suburban).
    # Scaled ×1000 to avoid near-zero float values in feature space.
    # ------------------------------------------------------------------
    gdf["grade_per_sqft"] = (gdf["grade"] / gdf["sqft_living"]) * 1000

    # ------------------------------------------------------------------
    # Feature 33: total_rooms_proxy
    # Simple additive count of bedrooms + bathrooms.
    # Lightweight room density proxy before spatial adjustment in Week 3.
    # ------------------------------------------------------------------
    gdf["total_rooms_proxy"] = gdf["bedrooms"] + gdf["bathrooms"]

    # ------------------------------------------------------------------
    # Feature 34: room_density
    # Total rooms (bed + bath) per 1000 sqft of living area.
    # Low = spacious layout. High = compact/dense room arrangement.
    # Scaled ×1000 for numeric stability.
    # ------------------------------------------------------------------
    gdf["room_density"] = (gdf["total_rooms_proxy"] / gdf["sqft_living"]) * 1000

    return gdf


def run(input_path: str, output_path: str) -> gpd.GeoDataFrame:
    """
    Load Category 2 output, apply Category 3 features, and save result.

    Parameters
    ----------
    input_path  : Path to spatial_proximity_features.parquet (Category 2 output)
    output_path : Path to write structural_ratio_features.parquet
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading Category 2 output from: {input_path}")
    gdf = gpd.read_parquet(input_path)

    # CRS guard — must be EPSG:2285 to be consistent with upstream pipeline
    expected_epsg = int(PROJECTED_CRS.split(":")[1])
    if gdf.crs is None or gdf.crs.to_epsg() != expected_epsg:
        raise ValueError(
            f"Expected CRS EPSG:{expected_epsg}, got {gdf.crs}. "
            "Ensure input was produced by the Category 2 pipeline."
        )

    print("Applying Category 3: Structural Ratios & Internal Composition...")
    gdf = add_structural_ratio_features(gdf)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_parquet(output_path, index=False)

    added_cols = [
        "bath_per_bed",            # 21
        "sqft_per_bed",            # 22
        "sqft_per_bath",           # 23
        "above_to_living_ratio",   # 24
        "basement_to_living_ratio",# 25
        "lot_to_living_ratio",     # 26
        "living_to_lot_pct",       # 27
        "floor_area_efficiency",   # 28
        "neighbor_living_ratio",   # 29
        "neighbor_lot_ratio",      # 30
        "grade_condition_score",   # 31
        "grade_per_sqft",          # 32
        "total_rooms_proxy",       # 33
        "room_density",            # 34
    ]

    print(f"\nCategory 3 complete! Added {len(added_cols)} features:")
    for i, col in enumerate(added_cols, start=21):
        print(f"    ├─ [{i}] {col}")

    print(f"\nOutput saved to: {output_path}")
    return gdf


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent

    input_path = BASE_DIR / "dataset" / "processed" / "spatial_proximity_features.parquet"
    output_path = BASE_DIR / "dataset" / "processed" / "structural_ratio_features.parquet"

    run(
        input_path=str(input_path),
        output_path=str(output_path),
    )
