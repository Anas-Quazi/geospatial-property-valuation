import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.model_selection import GroupKFold

def generate_spatial_splits(input_path: str, splits_output_path: str, group_column: str = "zipcode", n_splits: int = 5):
    """
    Reads the spatial data as an immutable, read-only resource.
    Generates an entirely independent validation mapping file to preserve data integrity.
    """
    # 1. Verify and read the invariant spatial data package
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Spatial data engine missing at: {input_path}")
        
    print("📖 Reading immutable spatial parquet file...")
    gdf = gpd.read_parquet(input_path)
    
    # BIG FIX: Force a clean, sequential RangeIndex (0, 1, 2, 3...)
    # This guarantees that physical row positions align perfectly with index labels.
    gdf = gdf.reset_index(drop=True)
    
    # Since 'id' is missing from your file, we explicitly use our guaranteed index
    gdf['generated_index'] = gdf.index
    id_column = 'generated_index'
        
    # Verify the spatial clustering parameter exists
    if group_column not in gdf.columns:
        raise KeyError(f"Specified spatial grouping column '{group_column}' missing from dataset.")

    # 2. Initialize GroupKFold Partitioning
    print(f"🔒 Allocating {n_splits}-Fold GroupKFold splits partitioned by administrative '{group_column}' blocks...")
    gkf = GroupKFold(n_splits=n_splits)
    
    # Create an empty array to house fold assignments corresponding exactly to the row sequences
    fold_assignments = pd.Series(index=gdf.index, dtype='int32')
    
    # Extract independent arrays for processing
    X = gdf[[id_column]].values
    y = gdf['price'].values if 'price' in gdf.columns else X
    groups = gdf[group_column].values
    
    # 3. Calculate folds while auditing for zero spatial data leakage
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        # Because we reset the index above, .iloc is now 100% safe to use
        fold_assignments.iloc[val_idx] = fold_idx
        
        # Mathematical verification of complete separation
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        
        print(f"    ├─ Fold {fold_idx}: Train Zones = {len(train_groups)} | Val Zones = {len(val_groups)} | Leakage Overlap = {len(overlap)}")
        assert len(overlap) == 0, f"FATAL ERROR: Spatial leakage detected in Fold {fold_idx}!"

    # 4. Construct the completely decoupled metadata mapping dataframe
    print("🛠️ Constructing independent validation indexing matrix...")
    
    splits_df = pd.DataFrame({
        "generated_index": gdf.index,
        "spatial_fold": fold_assignments
    })
    
    # 5. Serialize the validation mapping metadata to disk
    os.makedirs(os.path.dirname(splits_output_path), exist_ok=True)
    print(f"💾 Saving decoupled validation map asset to: {splits_output_path}")
    splits_df.to_parquet(splits_output_path, index=False)
    
    print("✅ Phase 4 complete! Spatial validation strategy locked down safely without data tampering.")

if __name__ == "__main__":
    # SCRIPT_DIR is project_root/src/data/
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # BASE_DIR steps up twice to hit project_root/
    BASE_DIR = SCRIPT_DIR.parent.parent

    # Absolute path targeting matching your directory layout
    input_spatial_data = BASE_DIR / "dataset" / "processed" / "kc_house_spatial.parquet"
    output_validation_splits = BASE_DIR / "dataset" / "processed" / "spatial_validation_splits.parquet"

    # Execute pipeline
    generate_spatial_splits(
        input_path=str(input_spatial_data),
        splits_output_path=str(output_validation_splits),
        group_column="zipcode",
        n_splits=5
    )