import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import point
from pathlib import Path

def convert_to_geodataframe(input_path: str, output_path: str):
    """
    Ingests a cleaned housing CSV, builds geographic geometries using Shapely,
    initializes a GeoDataFrame, and projects coordinates to local feet units.
    """

    #& Verify and load the cleaned dataset
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned dataset missing at: {input_path}")
        
    print("Loading cleaned tabular dataset...")
    df = pd.read_csv(input_path)
    
    #todo Vectorize coordinates into Shapely Points
    print("Converting scalar lat/long columns into geometric Points...")
    geometry = gpd.points_from_xy(df['long'], df['lat'])
    
    #* Initialize the GeoPandas GeoDataFrame with standard GPS CRS
    print("Initializing GeoDataFrame with WGS 84 (EPSG:4326)...")
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    #? Project to local Washington State Plane (North) coordinates (transform degree to physical feet)
    local_crs = "EPSG:2285"
    print(f"Re-projecting spatial engine to {local_crs} (Washington North - Feet)...")
    gdf = gdf.to_crs(local_crs)
    
    #! Extract projected X and Y coordinates for explicit model access (.x = east, .y = north)
    gdf['projected_x'] = gdf.geometry.x
    gdf['projected_y'] = gdf.geometry.y
    
    #^ Save the geo-spatial data package to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving geospatial dataset to: {output_path}")
    gdf.to_parquet(output_path, index=False)
    
    print("Phase 2 complete! Dataset structure successfully upgraded.")
    print(f"   Current Spatial Projection System: {gdf.crs}")
    print(f"   Sample Geometry Object: {gdf['geometry'].iloc[0]}")

if __name__ == "__main__":

    #! handle input and output path dynamically
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent

    candidate_inputs = [
        BASE_DIR / "dataset" / "kc_house_cleaned.csv",
        BASE_DIR / "kc_house_cleaned.csv",
        SCRIPT_DIR / "kc_house_cleaned.csv"
    ]

    input_file_path = None
    for path in candidate_inputs:
        if path.exists():
            input_file_path = path
            break

    if input_file_path is None:
        input_file_path = BASE_DIR / "dataset" / "kc_house_cleaned.csv"

    output_file_path = BASE_DIR / "dataset" / "processed" / "kc_house_spatial.parquet"

    #~ Execute the geospatial pipeline with resolved string paths
    convert_to_geodataframe(
        input_path=str(input_file_path),
        output_path=str(output_file_path)
    )