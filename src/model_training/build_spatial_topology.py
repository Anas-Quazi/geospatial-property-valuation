import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 1: Spatial Graph Topology Design
# ---------------------------------------------------------------------------
# Validates coordinate projections (EPSG:2285) and sets up the structural
# node/edge schema for spatial neighborhood modeling.
# Owner : Shais013
# Input : dataset/processed/kc_master_dataset_cleaned.parquet
# ---------------------------------------------------------------------------

TARGET_CRS         = "EPSG:2285"  # Washington State Plane North (Feet)
DEFAULT_K_NEIGHBORS = 10


def inspect_topology_data(path: Path):
    print(f"Loading data from: {path}")
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")

    # Column check — dataset uses 'lat' and 'long'
    if "lat" in df.columns and "long" in df.columns:
        print("✓ lat and long columns found.")
    else:
        raise KeyError("Expected columns 'lat' and 'long' not found in dataset.")

    # Convert to GeoDataFrame using lat/long (EPSG:4326 = WGS84 degrees)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["long"], df["lat"]), crs="EPSG:4326"
    )

    # Reproject to EPSG:2285 — flat physical grid in feet, accurate Euclidean distances
    gdf_projected = gdf.to_crs(TARGET_CRS)

    # Extract projected X, Y back into plain pandas columns for ML compatibility
    df["x_proj"] = gdf_projected.geometry.x
    df["y_proj"] = gdf_projected.geometry.y

    print(f"✓ Coordinates projected to {TARGET_CRS}.")
    print(f"  Sample X (Projected): {df['x_proj'].iloc[0]:,.2f}")
    print(f"  Sample Y (Projected): {df['y_proj'].iloc[0]:,.2f}")

    print("\n" + "=" * 50)
    print("  PHASE 1 TOPOLOGY DESIGN PARAMETERS LOCKED")
    print("=" * 50)
    print(f"  Total Nodes  (|V|)  : {len(df):,}")
    print(f"  Target K Neighbors  : {DEFAULT_K_NEIGHBORS}")
    print(f"  Expected Edges (|E|): {len(df) * DEFAULT_K_NEIGHBORS:,}")
    print("  Next Step           : Phase 2 — KNN Graph Construction")
    print("=" * 50)

    return df


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    candidate_paths = [
        script_dir / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet",
        script_dir.parent / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet",
        script_dir.parent.parent / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet",
        Path("dataset/processed/kc_master_dataset_cleaned.parquet"),
        Path("kc_master_dataset_cleaned.parquet"),
    ]

    input_path = next((p for p in candidate_paths if p.exists()), None)
    if input_path is None:
        raise FileNotFoundError("Could not find kc_master_dataset_cleaned.parquet in candidate paths.")

    inspect_topology_data(input_path)
