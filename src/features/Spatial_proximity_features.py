import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ---------------------------------------------------------------------------
# FIXED REFERENCE CONSTANTS
# ---------------------------------------------------------------------------
# These are real-world, static geographic facts — not derived from
# kc_house_data.csv or any train/validation split. They are the same for
# every row and every fold, so there is no data-leakage risk in using them.
# Source: publicly known city-center / campus coordinates (WGS84 lat/long).
# ---------------------------------------------------------------------------

HUBS_WGS84 = {
    "seattle":  {"lat": 47.6062, "lon": -122.3321},   # Downtown Seattle
    "bellevue": {"lat": 47.6101, "lon": -122.2015},   # Bellevue downtown core
    "redmond":  {"lat": 47.6740, "lon": -122.1215},   # Microsoft Redmond campus
}

KING_COUNTY_CENTER_WGS84 = {"lat": 47.4670, "lon": -121.8330}  # King County center

# A small set of fixed shoreline / lake reference points (WGS84 lat/long).
# dist_to_nearest_coast / dist_to_nearest_lake take the MINIMUM distance
# from a house to any point in the relevant set — a lightweight stand-in
# for a full shoreline polygon, with no external shapefile dependency.

PUGET_SOUND_POINTS_WGS84 = [
    {"lat": 47.6050, "lon": -122.3800},  # Elliott Bay / downtown waterfront
    {"lat": 47.6870, "lon": -122.4020},  # Shilshole / Ballard
    {"lat": 47.5000, "lon": -122.4300},  # Vashon Island facing shore
    {"lat": 47.3090, "lon": -122.3350},  # Des Moines / south sound
    {"lat": 47.7580, "lon": -122.3960},  # Richmond Beach / north county
]

LAKE_POINTS_WGS84 = [
    {"lat": 47.6205, "lon": -122.2529},  # Lake Washington (central)
    {"lat": 47.5480, "lon": -122.2610},  # Lake Washington (south, Renton end)
    {"lat": 47.6870, "lon": -122.2470},  # Lake Washington (north, Kenmore end)
    {"lat": 47.6018, "lon": -122.0844},  # Lake Sammamish (central)
]

# CRS used throughout the project (matches kc_house_spatial.parquet):
# NAD83 / Washington North (EPSG:2285), units = US survey feet.
PROJECTED_CRS = "EPSG:2285"
WGS84_CRS = "EPSG:4326"


def _points_to_projected_xy(points_wgs84, crs=PROJECTED_CRS):
    """
    Convert a list of {"lat":, "lon":} dicts into a list of (x, y) tuples
    in the project's projected CRS (feet), so distances come out in feet
    and stay consistent with projected_x / projected_y already in the data.
    """
    lats = [p["lat"] for p in points_wgs84]
    lons = [p["lon"] for p in points_wgs84]
    gseries = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs=WGS84_CRS)
    projected = gseries.to_crs(crs)
    return list(zip(projected.x.values, projected.y.values))


def _min_distance_to_points(x, y, ref_xy):
    """
    Given arrays of house x/y coordinates and a list of reference (x, y)
    points, return the minimum Euclidean distance (in feet) from each
    house to the closest reference point.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    min_dist = np.full(x.shape, np.inf)
    for rx, ry in ref_xy:
        d = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
        min_dist = np.minimum(min_dist, d)
    return min_dist


def add_spatial_proximity_features(gdf):
    """
    Category 2: Spatial Proximity & Accessibility (12 features).

    Reads the spatial parquet (already projected to EPSG:2285), adds the
    12 Category 2 features as pure, per-row deterministic transforms
    against fixed external reference points, and writes the result to a
    new parquet file. Does NOT touch fold assignment or any other
    category's features — scope is strictly Category 2.
    """

    if "projected_x" not in gdf.columns or "projected_y" not in gdf.columns:
        raise KeyError(
            "Expected 'projected_x' / 'projected_y' columns (EPSG:2285) "
            "not found — run geospatial_conversion.py first."
        )

    
    # Guard against silent unit-mismatch: if the input file's CRS is not
    # EPSG:2285, distances computed against our EPSG:2285 reference points
    # would be silently wrong (mixed units) rather than erroring. Fail loud.
    expected_epsg = int(PROJECTED_CRS.split(":")[1])
    if gdf.crs is None or gdf.crs.to_epsg() != expected_epsg:
        raise ValueError(
            f"Expected input CRS EPSG:{expected_epsg}, but got {gdf.crs}. "
            "Distances would be computed in mismatched units — aborting "
            "rather than silently producing incorrect feature values."
        )

    # Guard against silent NaN propagation from bad upstream geocoding.
    n_null_xy = int(gdf["projected_x"].isna().sum() + gdf["projected_y"].isna().sum())
    if n_null_xy > 0:
        raise ValueError(
            f"Found {n_null_xy} null value(s) in projected_x/projected_y. "
            "Fix upstream geocoding before computing distance features."
        )

    x = gdf["projected_x"].values
    y = gdf["projected_y"].values

   # -----------------------------------------------------------------
    # Precompute projected (x, y) for every fixed reference point ONCE.
    # -----------------------------------------------------------------
    print(" Projecting fixed reference points to EPSG:2285...")
    hub_xy = {
        name: _points_to_projected_xy([coords])[0]
        for name, coords in HUBS_WGS84.items()
    }
    county_center_xy = _points_to_projected_xy([KING_COUNTY_CENTER_WGS84])[0]
    coast_xy = _points_to_projected_xy(PUGET_SOUND_POINTS_WGS84)
    lake_xy = _points_to_projected_xy(LAKE_POINTS_WGS84)

    # -----------------------------------------------------------------
    # 9-14: Hub distances (Seattle, Bellevue, Redmond) + log versions
    # -----------------------------------------------------------------
    print("Computing hub distances (Seattle / Bellevue / Redmond)...")
    for hub_name in ("seattle", "bellevue", "redmond"):
        hx, hy = hub_xy[hub_name]
        dist = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
        gdf[f"dist_to_{hub_name}"] = dist
        gdf[f"log_dist_to_{hub_name}"] = np.log1p(dist)  # ln(d + 1)

    
    # -----------------------------------------------------------------
    # 15-16: Coastline & lake proximity (minimum distance to point set)
    # -----------------------------------------------------------------
    print("Computing coastline & lake proximity...")
    gdf["dist_to_nearest_coast"] = _min_distance_to_points(x, y, coast_xy)
    gdf["dist_to_nearest_lake"] = _min_distance_to_points(x, y, lake_xy)

   # -----------------------------------------------------------------
    # 17-20: Geographic coordinates & projections
    # -----------------------------------------------------------------
    print("Adding coordinate & projection features...")
    gdf["x_coords"] = x
    gdf["y_coords"] = y

    # NOTE: safe for King County (long is always ~-122, never 0), but this
    # division has no zero-guard. If reused for a region where longitude
    # can be 0 or very small, this would silently produce inf/huge values.
    gdf["lat_lon_ratio"] = gdf["lat"] / gdf["long"]

    cx, cy = county_center_xy
    gdf["radial_dist_origin"] = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


    added_cols = [
        "dist_to_seattle", "log_dist_to_seattle",
        "dist_to_bellevue", "log_dist_to_bellevue",
        "dist_to_redmond", "log_dist_to_redmond",
        "dist_to_nearest_coast", "dist_to_nearest_lake",
        "x_coords", "y_coords", "lat_lon_ratio", "radial_dist_origin",
    ]
    print(f"Category 2 complete! Added {len(added_cols)} features:")
    for c in added_cols:
        print(f"    ├─ {c}")

    return gdf


if __name__ == "__main__":
    # SCRIPT_DIR is project_root/src/data/
    SCRIPT_DIR = Path(__file__).resolve().parent

    # BASE_DIR steps up twice to hit project_root/
    BASE_DIR = SCRIPT_DIR.parent.parent

    input_spatial_data = BASE_DIR / "dataset" / "processed" / "kc_house_spatial.parquet"
    df = pd.read_parquet(input_spatial_data)
    add_spatial_proximity_features(df)
