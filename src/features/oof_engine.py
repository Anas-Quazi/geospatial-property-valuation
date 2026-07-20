import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
 
#! calc nearest neighbor for zipcode(s)
def _knn_weighted_lookup(query_xy, ref_xy, ref_values, k=5, eps=1e-6):
    """Inverse-distance-weighted average of ref_values at the k nearest ref_xy points."""

    tree = cKDTree(ref_xy)
    k = min(k, len(ref_xy))
    dist, idx = tree.query(query_xy, k=k)

    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    w = 1.0 / (dist + eps)
    w = w / w.sum(axis=1, keepdims=True)
    vals = ref_values[idx]  

    return (vals * w).sum(axis=1)
 
 
def add_oof_features(df, block_size=2000.0, k_neighbors=5):
    """
    block_size: grid cell size in the same units as projected_x/projected_y
                (Washington State Plane, US feet -> 2000 ft ~ 0.38 mi block)
    k_neighbors: how many nearest train zips/blocks to blend when the exact
                 zip/block wasn't seen in this fold's training data
    """

    df = df.copy()
    #* required features
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    df['age_at_sale'] = df['sale_year'] - df['yr_built']
 
    #todo spatial grid block id, built from projected coords
    df['block_x'] = (df['projected_x'] // block_size).astype(int)
    df['block_y'] = (df['projected_y'] // block_size).astype(int)
    df['block_id'] = df['block_x'].astype(str) + '_' + df['block_y'].astype(str)
 
    oof_cols = [
        'oof_target_enc_zip_price_sqft', 'oof_target_enc_zip_price',
        'relative_sqft_to_zip_median', 'relative_grade_to_zip_median',
        'relative_price_sqft_to_zip_median', 'zip_density_indicator',
        'relative_age_to_zip_median', 'oof_target_enc_block_price'
    ]
    for col in oof_cols:
        df[col] = np.nan
 
    #? global fallback
    g_psqft = df['price_per_sqft'].median()
    g_price = df['price'].median()
    g_sqft = df['sqft_living'].median()
    g_grade = df['grade'].median()
    g_age = df['age_at_sale'].median()
 
    folds = sorted(df['fold'].dropna().unique())
 
    for fold in folds:
        train_idx = df[df['fold'] != fold].index
        val_idx = df[df['fold'] == fold].index
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
 
        #^ zip-level stats (train only)
        zip_stats = train_df.groupby('zipcode').agg(
            zip_mean_psqft=('price_per_sqft', 'mean'),
            zip_mean_price=('price', 'mean'),
            zip_med_sqft=('sqft_living', 'median'),
            zip_med_grade=('grade', 'median'),
            zip_med_age=('age_at_sale', 'median'),
            zip_count=('price', 'size'),
            zip_x=('projected_x', 'mean'),
            zip_y=('projected_y', 'mean'),
        )
 
        zip_xy = zip_stats[['zip_x', 'zip_y']].to_numpy()
        val_xy = val_df[['projected_x', 'projected_y']].to_numpy()
 
        def zip_feature(col_name, fallback):
            direct = val_df['zipcode'].map(zip_stats[col_name])
            missing = direct.isna()
            if missing.any() and len(zip_stats) > 0:
                knn_vals = _knn_weighted_lookup(
                    val_xy[missing.values], zip_xy,
                    zip_stats[col_name].to_numpy(), k=k_neighbors
                )
                direct.loc[missing] = knn_vals
            return direct.fillna(fallback)
 
        oof_psqft = zip_feature('zip_mean_psqft', g_psqft)
        oof_price = zip_feature('zip_mean_price', g_price)
        zip_sqft_med = zip_feature('zip_med_sqft', g_sqft)
        zip_grade_med = zip_feature('zip_med_grade', g_grade)
        zip_age_med = zip_feature('zip_med_age', g_age)
        zip_count = zip_feature('zip_count', 0.0)
 
        df.loc[val_idx, 'oof_target_enc_zip_price_sqft'] = oof_psqft.values
        df.loc[val_idx, 'oof_target_enc_zip_price'] = oof_price.values
        df.loc[val_idx, 'relative_sqft_to_zip_median'] = (val_df['sqft_living'] / zip_sqft_med).values
        df.loc[val_idx, 'relative_grade_to_zip_median'] = (val_df['grade'] / zip_grade_med).values
        df.loc[val_idx, 'relative_price_sqft_to_zip_median'] = (val_df['price_per_sqft'] / oof_psqft).values
        df.loc[val_idx, 'relative_age_to_zip_median'] = (val_df['age_at_sale'] / zip_age_med.replace(0, np.nan)).fillna(1.0).values
        df.loc[val_idx, 'zip_density_indicator'] = zip_count.values
 
        #~ block-level price (train only, spatial fallback)
        block_stats = train_df.groupby('block_id').agg(
            block_mean_price=('price', 'mean'),
            block_x=('projected_x', 'mean'),
            block_y=('projected_y', 'mean'),
        )
        block_xy = block_stats[['block_x', 'block_y']].to_numpy()
 
        direct_block = val_df['block_id'].map(block_stats['block_mean_price'])
        missing_block = direct_block.isna()
        if missing_block.any() and len(block_stats) > 0:
            knn_block = _knn_weighted_lookup(
                val_xy[missing_block.values], block_xy,
                block_stats['block_mean_price'].to_numpy(), k=k_neighbors
            )
            direct_block.loc[missing_block] = knn_block
        df.loc[val_idx, 'oof_target_enc_block_price'] = direct_block.fillna(g_price).values
 
    df = df.drop(columns=['block_x', 'block_y'])
 
    if df[oof_cols].isnull().any().any():
        missing_count = df[oof_cols].isnull().sum().sum()
        raise ValueError(f"STILL HAVE {missing_count} MISSING VALUES. CHECK MAPPING LOGIC.")
 
    return df

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    candidates_paths = [
        BASE_DIR / "dataset" / "processed" / "master_static_features.parquet",
        BASE_DIR.parent / "dataset" / "processed" / "master_static_features.parquet",
    ]

    dataset_input_path = next((p for p in candidates_paths if p.exists()), None)
    
    if dataset_input_path is None:
        raise FileNotFoundError("master_static_features.parquet not found in any expected location")

    df = pd.read_parquet(dataset_input_path)
    df = add_oof_features(df)

    output_path = output_path = BASE_DIR / "dataset" / "processed" / "kc_master_dataset.parquet"
    df.to_parquet(output_path, index=False)

