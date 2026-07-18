import pandas as pd
import geopandas as gpd
from pathlib import Path

#& import transformed functions
from temporal_features import add_temporal_decay_features
from Spatial_proximity_features import add_spatial_proximity_features
from structural_ratio_features import add_structural_ratio_features

def run_master_merge(input_path : str, sv_data_path : str, output_path : str):
    
    #~ load raw spatial dataset
    spatial_df = gpd.read_parquet(input_path)
    
    #^ Load validation splits data
    val_splits_df = pd.read_parquet(sv_data_path)

    #! reanme columns before merge
    val_splits_df = val_splits_df.rename(columns={'generated_index' : 'id', 'spatial_fold' : 'fold'})

    #! handle id in orignal dataset
    spatial_df['id'] = range(len(spatial_df))
    
    #* merge dataset with validation folds on "id"
    df = spatial_df.merge(val_splits_df[['id', 'fold']], on='id', how='left')
    
    #todo Apply built features on dataframe
    print("Applying transformations...")
    df = add_temporal_decay_features(df)
    df = add_spatial_proximity_features(df)
    df = add_structural_ratio_features(df)
    
    #? save master static file
    df.to_parquet(output_path, index=False)
    print(f"Master static dataset (with folds) saved to {output_path}")

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    spatial_candidates = [
        BASE_DIR / "dataset" / "processed" / "kc_house_spatial.parquet",
        BASE_DIR.parent / "dataset" / "processed" / "kc_house_spatial.parquet",
    ]
    sv_candidates = [
        BASE_DIR / "dataset" / "processed" / "spatial_validation_splits.parquet",
        BASE_DIR.parent / "dataset" / "processed" / "spatial_validation_splits.parquet",
    ]
    
    dataset_input_path = next((p for p in spatial_candidates if p.exists()), None)
    sv_dataset_path = next((p for p in sv_candidates if p.exists()), None)
    
    if dataset_input_path is None:
        raise FileNotFoundError("kc_house_spatial.parquet not found in any expected location")
    if sv_dataset_path is None:
        raise FileNotFoundError("spatial_validation_splits.parquet not found in any expected location")
    
    output_path = BASE_DIR / "dataset" / "processed" / "master_static_features.parquet"

    run_master_merge(dataset_input_path, sv_dataset_path, output_path)