import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Fixed reference points (WGS84 lat/lon). Static real-world locations,
# not derived from the dataset, so no leakage risk across folds.
HUBS_WGS84 = {
    "seattle":  {"lat": 47.6062, "lon": -122.3321},
    "bellevue": {"lat": 47.6101, "lon": -122.2015},
    "redmond":  {"lat": 47.6740, "lon": -122.1215},  # Microsoft campus
}

KING_COUNTY_CENTER_WGS84 = {"lat": 47.4670, "lon": -121.8330}
# Hardcoded on purpose, not computed from house coordinates — a centroid
# derived from the dataset would differ across folds/samples and leak.

# Small hardcoded point sets for coast/lake distance (min distance to
# any point in the set). Avoids pulling in a full shoreline shapefile.
PUGET_SOUND_POINTS_WGS84 = [
    {"lat": 47.6050, "lon": -122.3800},  # Elliott Bay
    {"lat": 47.6870, "lon": -122.4020},  # Shilshole
    {"lat": 47.5000, "lon": -122.4300},  # Vashon Island
    {"lat": 47.3090, "lon": -122.3350},  # Des Moines
    {"lat": 47.7580, "lon": -122.3960},  # Richmond Beach
]

LAKE_POINTS_WGS84 = [
    {"lat": 47.6205, "lon": -122.2529},  # Lake Washington, central
    {"lat": 47.5480, "lon": -122.2610},  # Lake Washington, Renton end
    {"lat": 47.6870, "lon": -122.2470},  # Lake Washington, Kenmore end
    {"lat": 47.6018, "lon": -122.0844},  # Lake Sammamish
]

PROJECTED_CRS = "EPSG:2285"  # matches kc_house_spatial.parquet (feet)
WGS84_CRS = "EPSG:4326"


def _points_to_projected_xy(points_wgs84, crs=PROJECTED_CRS):
    """Project a list of {lat, lon} dicts to (x, y) in EPSG:2285 feet."""
    lats = [p["lat"] for p in points_wgs84]
    lons = [p["lon"] for p in points_wgs84]
    gseries = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs=WGS84_CRS)
    projected = gseries.to_crs(crs)
    return list(zip(projected.x.values, projected.y.values))


def _min_distance_to_points(x, y, ref_xy):
    """Min distance (feet) from each house to the closest ref point."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    min_dist = np.full(x.shape, np.inf)
    for rx, ry in ref_xy:
        d = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
        min_dist = np.minimum(min_dist, d)
    return min_dist


def add_spatial_proximity_features(gdf):
    """Category 2: Spatial Proximity & Accessibility (12 features)."""
    # All 12 features are per-row transforms against fixed points above —
    # no dependency on price or fold, so fold assignment isn't used here.

    if "projected_x" not in gdf.columns or "projected_y" not in gdf.columns:
        raise KeyError(
            "projected_x/projected_y not found — run geospatial_conversion.py first."
        )

    # CRS must match, or distances end up in mixed units.
    expected_epsg = int(PROJECTED_CRS.split(":")[1])
    if gdf.crs is None or gdf.crs.to_epsg() != expected_epsg:
        raise ValueError(f"Expected CRS EPSG:{expected_epsg}, got {gdf.crs}.")

    n_null_xy = int(gdf["projected_x"].isna().sum() + gdf["projected_y"].isna().sum())
    if n_null_xy > 0:
        raise ValueError(f"Found {n_null_xy} null value(s) in projected_x/projected_y.")

    x = gdf["projected_x"].values
    y = gdf["projected_y"].values

    print("Projecting reference points to EPSG:2285...")
    hub_xy = {
        name: _points_to_projected_xy([coords])[0]
        for name, coords in HUBS_WGS84.items()
    }
    county_center_xy = _points_to_projected_xy([KING_COUNTY_CENTER_WGS84])[0]
    coast_xy = _points_to_projected_xy(PUGET_SOUND_POINTS_WGS84)
    lake_xy = _points_to_projected_xy(LAKE_POINTS_WGS84)

    print("Computing hub distances...")
    for hub_name in ("seattle", "bellevue", "redmond"):
        hx, hy = hub_xy[hub_name]
        dist = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
        gdf[f"dist_to_{hub_name}"] = dist
        gdf[f"log_dist_to_{hub_name}"] = np.log1p(dist)

    print("Computing coastline & lake proximity...")
    gdf["dist_to_nearest_coast"] = _min_distance_to_points(x, y, coast_xy)
    gdf["dist_to_nearest_lake"] = _min_distance_to_points(x, y, lake_xy)

    print("Adding coordinate features...")
    gdf["x_coords"] = x
    gdf["y_coords"] = y
    gdf["lat_lon_ratio"] = gdf["lat"] / gdf["long"]  # long is never 0 in this dataset

    cx, cy = county_center_xy
    gdf["radial_dist_origin"] = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    added_cols = [
        "dist_to_seattle", "log_dist_to_seattle",
        "dist_to_bellevue", "log_dist_to_bellevue",
        "dist_to_redmond", "log_dist_to_redmond",
        "dist_to_nearest_coast", "dist_to_nearest_lake",
        "x_coords", "y_coords", "lat_lon_ratio", "radial_dist_origin",
    ]
    print(f"Category 2 done. Added {len(added_cols)} features.")

    return gdf


if __name__ == "__main__":
    # Legacy standalone entry point — kept for quick manual smoke-testing
    # of this module in isolation. Not part of the main pipeline; that
    # runs through master_merge.py, which calls add_spatial_proximity_features()
    # directly on an in-memory dataframe. Running this file alone just
    # computes the features and discards them (no file is saved).
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent

    input_spatial_data = BASE_DIR / "dataset" / "processed" / "kc_house_spatial.parquet"
    df = gpd.read_parquet(input_spatial_data)
    add_spatial_proximity_features(df)