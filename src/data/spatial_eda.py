import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster

def generate_spatial_eda_map(input_path: str, output_html_path: str):
    """
    Ingests spatial Parquet data, reverses coordinate projection to global GPS degrees,
    and exports a fully interactive dual-layer price heatmap and marker cluster map.
    """
    # 1. Verify and ingest the spatial parquet data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Spatial parquet file missing at: {input_path}")
        
    print("Loading spatial Parquet dataset...")
    gdf = gpd.read_parquet(input_path)
    
    # 2. Convert coordinate reference system back to standard GPS decimal degrees
    # Web mapping libraries (Folium/Leaflet) strictly require EPSG:4326 (WGS 84)
    if gdf.crs != "EPSG:4326":
        print("Converting spatial coordinates back to global GPS degrees (EPSG:4326)...")
        gdf_gps = gdf.to_crs("EPSG:4326")
    else:
        gdf_gps = gdf.copy()
        
    # Extract raw GPS latitude and longitude values from the Shapely Point objects
    gdf_gps['lat_gps'] = gdf_gps.geometry.y
    gdf_gps['long_gps'] = gdf_gps.geometry.x

    # 3. Initialize the core Folium Map centered on the data's geographic median
    center_lat = gdf_gps['lat_gps'].median()
    center_long = gdf_gps['long_gps'].median()
    
    print(f"Map canvas anchored over King County center: [{center_lat:.4f}, {center_long:.4f}]")
    m = folium.Map(location=[center_lat, center_long], zoom_start=10, tiles="OpenStreetMap")

    # 4. Construct Layer 1: Property Price Heatmap
    print("Rendering dynamic pricing density heatmap...")
    # We use price per square foot of living space to capture localized geographic value premiums
    gdf_gps['price_per_sqft'] = gdf_gps['price'] / gdf_gps['sqft_living']
    
    # Structure data format required by Folium HeatMap: [[lat, long, weight], ...]
    heatmap_data = gdf_gps[['lat_gps', 'long_gps', 'price_per_sqft']].dropna().values.tolist()
    
    # Add the heatmap layer to the map canvas
    HeatMap(
        data=heatmap_data, 
        radius=12, 
        blur=8, 
        max_zoom=13,
        name="Price Intensity Heatmap"
    ).add_to(m)

    # 5. Construct Layer 2: Luxury Hotspot Markers (Top 2% Highest Prices)
    print("Building cluster layers for luxury property hotspots...")
    price_threshold = gdf_gps['price'].quantile(0.98)
    luxury_homes = gdf_gps[gdf_gps['price'] >= price_threshold]
    
    marker_cluster = MarkerCluster(name="Luxury Properties (Top 2%)").add_to(m)
    
    for idx, row in luxury_homes.iterrows():
        # Compose a clean, readable text popup box for individual properties
        popup_html = f"""
        <div style='font-family: Arial, sans-serif; min-width: 150px;'>
            <h4 style='margin: 0 0 5px 0; color: #2c3e50;'>House Sale Profile</h4>
            <hr style='margin: 5px 0;'>
            <b>Price:</b> ${int(row['price']):,}<br>
            <b>Bedrooms:</b> {int(row['bedrooms'])}<br>
            <b>Bathrooms:</b> {row['bathrooms']}<br>
            <b>Grade Rating:</b> {int(row['grade'])}/13<br>
            <b>Renovated Year:</b> {int(row['yr_renovated']) if row['yr_renovated'] > 0 else 'Never'}
        </div>
        """
        folium.Marker(
            location=[row['lat_gps'], row['long_gps']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="purple", icon="home", prefix="fa")
        ).add_to(marker_cluster)

    # 6. Finalize Map Configuration and Save Asset
    folium.LayerControl().add_to(m)  # Allows turning layers on/off interactively
    
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    print(f"Exporting interactive HTML map asset to: {output_html_path}")
    m.save(output_html_path)
    
    print("Phase 3 complete! Interactive Spatial EDA dashboard successfully generated.")

if __name__ == "__main__":
    # Dynamically resolve file directories using our base root engine pathing logic
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent

    # Find the processed spatial parquet package
    input_spatial_data = BASE_DIR / "dataset" / "processed" / "kc_house_spatial.parquet"
    
    # Enforce saving the visualization file inside a new local reports folder
    output_html_report = BASE_DIR / "dataset" / "processed" / "spatial_eda_heatmap.html"

    # Run the visualization pipeline
    generate_spatial_eda_map(
        input_path=str(input_spatial_data),
        output_html_path=str(output_html_report)
    )