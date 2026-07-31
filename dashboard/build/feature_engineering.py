"""
Replicates the feature engineering pipeline from src/features/*.py so that
brand-new (never-seen) property inputs can be scored by the real trained
XGBoost model (models/xgb_baseline_model.json), using the same 60 engineered
features it was trained on.

Reference stats (zip aggregates, block aggregates, medians) are computed
once at startup from the real dataset (kc_master_dataset_cleaned.parquet)
and reused for every request.
"""
import numpy as np
import pandas as pd

FEATURE_ORDER = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
    'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
    'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15',
    'sale_year', 'sale_month', 'is_renovated', 'house_age', 'projected_x',
    'projected_y', 'years_since_renovated', 'yrs_between_built_and_renovated',
    'material_decay_rate', 'renovated_decay_rate', 'dist_to_seattle',
    'log_dist_to_seattle', 'dist_to_bellevue', 'log_dist_to_bellevue',
    'dist_to_redmond', 'log_dist_to_redmond', 'dist_to_nearest_coast',
    'dist_to_nearest_lake', 'x_coords', 'y_coords', 'lat_lon_ratio',
    'radial_dist_origin', 'land_to_structure_ratio', 'sqft_non_living',
    'avg_room_size', 'bed_bath_ratio', 'sqft_living_per_floor', 'is_mansion',
    'luxury_score', 'has_basement', 'basement_to_living_ratio',
    'above_to_living_ratio', 'living_to_lot15_ratio', 'lot_to_lot15_ratio',
    'total_rooms', 'grade_to_condition_ratio', 'age_at_sale',
    'relative_sqft_to_zip_median', 'relative_grade_to_zip_median',
    'zip_density_indicator', 'relative_age_to_zip_median',
    'oof_target_enc_block_price',
]

HUBS_WGS84 = {
    "seattle": {"lat": 47.6062, "lon": -122.3321},
    "bellevue": {"lat": 47.6101, "lon": -122.2015},
    "redmond": {"lat": 47.6740, "lon": -122.1215},
}
PUGET_SOUND_POINTS_WGS84 = [
    (47.6050, -122.3800), (47.6870, -122.4020), (47.5000, -122.4300),
    (47.3090, -122.3350), (47.7580, -122.3960),
]
LAKE_POINTS_WGS84 = [
    (47.6205, -122.2529), (47.5480, -122.2610),
    (47.6870, -122.2470), (47.6018, -122.0844),
]
BLOCK_SIZE = 2000.0


class ReferenceContext:
    """Precomputed lookup tables + affine lat/long->projected-feet transform,
    fit once from the real dataset (mean error ~30ft vs the true EPSG:2285
    projection used in the original pipeline -- negligible at city scale)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        lat = df['lat'].values
        lon = df['long'].values
        X = np.column_stack([lat, lon, lat * lon, np.ones_like(lat)])
        Y = np.column_stack([df['projected_x'].values, df['projected_y'].values])
        self.affine_coef, *_ = np.linalg.lstsq(X, Y, rcond=None)

        self.hub_xy = {name: self._project(v['lat'], v['lon']) for name, v in HUBS_WGS84.items()}
        self.coast_xy = [self._project(la, lo) for la, lo in PUGET_SOUND_POINTS_WGS84]
        self.lake_xy = [self._project(la, lo) for la, lo in LAKE_POINTS_WGS84]

        # Global fallbacks
        self.g_psqft = (df['price'] / df['sqft_living']).median()
        self.g_price = df['price'].median()
        self.g_sqft = df['sqft_living'].median()
        self.g_grade = df['grade'].median()
        self.g_age = (df['sale_year'] - df['yr_built']).median()

        # Zip-level aggregates (whole dataset -- serving time, no fold split needed)
        tmp = df.copy()
        tmp['price_per_sqft'] = tmp['price'] / tmp['sqft_living']
        tmp['age_at_sale'] = tmp['sale_year'] - tmp['yr_built']
        self.zip_stats = tmp.groupby('zipcode').agg(
            zip_mean_psqft=('price_per_sqft', 'mean'),
            zip_mean_price=('price', 'mean'),
            zip_med_sqft=('sqft_living', 'median'),
            zip_med_grade=('grade', 'median'),
            zip_med_age=('age_at_sale', 'median'),
            zip_count=('price', 'size'),
        )
        self.zip_xy = df.groupby('zipcode')[['projected_x', 'projected_y']].mean()

        # Block-level aggregates
        bx = (df['projected_x'] // BLOCK_SIZE).astype(int).astype(str)
        by = (df['projected_y'] // BLOCK_SIZE).astype(int).astype(str)
        tmp['block_id'] = bx + '_' + by
        self.block_stats = tmp.groupby('block_id').agg(
            block_mean_price=('price', 'mean'),
        )
        self.block_xy = tmp.groupby('block_id')[['projected_x', 'projected_y']].mean()

    def _project(self, lat, lon):
        row = np.array([lat, lon, lat * lon, 1.0])
        xy = row @ self.affine_coef
        return float(xy[0]), float(xy[1])

    def _nearest_zip_fallback(self, col, px, py, k=5):
        xy = self.zip_xy.to_numpy()
        d = np.sqrt((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2)
        k = min(k, len(d))
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-6)
        w = w / w.sum()
        vals = self.zip_stats[col].to_numpy()[idx]
        return float((vals * w).sum())

    def _nearest_block_fallback(self, px, py, k=5):
        xy = self.block_xy.to_numpy()
        d = np.sqrt((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2)
        k = min(k, len(d))
        idx = np.argsort(d)[:k]
        w = 1.0 / (d[idx] + 1e-6)
        w = w / w.sum()
        vals = self.block_stats['block_mean_price'].to_numpy()[idx]
        return float((vals * w).sum())

    def build_features(self, raw: dict, neutralize_location=False) -> dict:
        """raw must contain: lat, long, sqft_living, sqft_lot, bedrooms, bathrooms,
        floors, waterfront, view, condition, grade, sqft_above, sqft_basement,
        yr_built, yr_renovated, zipcode(optional), sqft_living15(optional),
        sqft_lot15(optional), sale_year(optional), sale_month(optional)."""
        f = {}
        f['bedrooms'] = raw['bedrooms']
        f['bathrooms'] = raw['bathrooms']
        f['sqft_living'] = raw['sqft_living']
        f['sqft_lot'] = raw['sqft_lot']
        f['floors'] = raw.get('floors', 1.0)
        f['waterfront'] = raw.get('waterfront', 0)
        f['view'] = raw.get('view', 0)
        f['condition'] = raw['condition']
        f['grade'] = raw['grade']
        f['sqft_above'] = raw.get('sqft_above', raw['sqft_living'] - raw.get('sqft_basement', 0))
        f['sqft_basement'] = raw.get('sqft_basement', 0)
        f['yr_built'] = raw['yr_built']
        f['yr_renovated'] = raw.get('yr_renovated', 0)
        f['sqft_living15'] = raw.get('sqft_living15', f['sqft_living'])
        f['sqft_lot15'] = raw.get('sqft_lot15', f['sqft_lot'])
        f['sale_year'] = raw.get('sale_year', 2026)
        f['sale_month'] = raw.get('sale_month', 7)
        f['is_renovated'] = 1 if f['yr_renovated'] and f['yr_renovated'] > 0 else 0
        f['house_age'] = f['sale_year'] - f['yr_built']

        lat = raw['lat']
        lon = raw['long']
        if neutralize_location:
            # Ablation: replace location with the county-wide average location,
            # isolating the model's structural (non-spatial) valuation.
            lat = self.df['lat'].mean()
            lon = self.df['long'].mean()
        f['lat'] = lat
        f['long'] = lon

        # nearest zipcode by projected distance if not supplied
        px, py = self._project(lat, lon)
        if raw.get('zipcode') and not neutralize_location:
            zipcode = raw['zipcode']
        else:
            zxy = self.zip_xy.to_numpy()
            d = np.sqrt((zxy[:, 0] - px) ** 2 + (zxy[:, 1] - py) ** 2)
            zipcode = self.zip_xy.index[int(np.argmin(d))]
        f['zipcode'] = zipcode

        f['projected_x'] = px
        f['projected_y'] = py

        f['years_since_renovated'] = (f['sale_year'] - f['yr_renovated']) if f['is_renovated'] else f['house_age']
        f['yrs_between_built_and_renovated'] = (f['yr_renovated'] - f['yr_built']) if f['is_renovated'] else 0
        f['material_decay_rate'] = float(np.exp(-0.05 * f['house_age']))
        f['renovated_decay_rate'] = float(np.exp(-0.05 * f['years_since_renovated']))

        for name in ("seattle", "bellevue", "redmond"):
            hx, hy = self.hub_xy[name]
            dist = float(np.sqrt((px - hx) ** 2 + (py - hy) ** 2))
            f[f'dist_to_{name}'] = dist
            f[f'log_dist_to_{name}'] = float(np.log1p(dist))

        f['dist_to_nearest_coast'] = float(min(np.sqrt((px - cx) ** 2 + (py - cy) ** 2) for cx, cy in self.coast_xy))
        f['dist_to_nearest_lake'] = float(min(np.sqrt((px - lx) ** 2 + (py - ly) ** 2) for lx, ly in self.lake_xy))
        f['x_coords'] = px
        f['y_coords'] = py
        f['lat_lon_ratio'] = lat / lon
        cx, cy = self._project(47.4670, -121.8330)
        f['radial_dist_origin'] = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2))

        f['land_to_structure_ratio'] = f['sqft_living'] / f['sqft_lot']
        f['sqft_non_living'] = f['sqft_lot'] - f['sqft_living']
        f['avg_room_size'] = f['sqft_living'] / (f['bedrooms'] + f['bathrooms'])
        f['bed_bath_ratio'] = f['bedrooms'] / f['bathrooms']
        f['sqft_living_per_floor'] = f['sqft_living'] / f['floors']
        f['is_mansion'] = int(f['sqft_living'] > 4000 and f['grade'] >= 10)
        f['luxury_score'] = f['grade'] * f['condition']
        f['has_basement'] = int(f['sqft_basement'] > 0)
        f['basement_to_living_ratio'] = f['sqft_basement'] / f['sqft_living']
        f['above_to_living_ratio'] = f['sqft_above'] / f['sqft_living']
        f['living_to_lot15_ratio'] = f['sqft_living'] / f['sqft_living15']
        f['lot_to_lot15_ratio'] = f['sqft_lot'] / f['sqft_lot15']
        f['total_rooms'] = f['bedrooms'] + f['bathrooms']
        f['grade_to_condition_ratio'] = f['grade'] / f['condition']
        f['age_at_sale'] = f['sale_year'] - f['yr_built']

        if neutralize_location:
            f['relative_sqft_to_zip_median'] = 1.0
            f['relative_grade_to_zip_median'] = 1.0
            f['zip_density_indicator'] = float(self.zip_stats['zip_count'].median())
            f['relative_age_to_zip_median'] = 1.0
            f['oof_target_enc_block_price'] = self.g_price
        else:
            if zipcode in self.zip_stats.index:
                zs = self.zip_stats.loc[zipcode]
                zip_sqft_med = zs['zip_med_sqft']
                zip_grade_med = zs['zip_med_grade']
                zip_age_med = zs['zip_med_age']
                zip_count = zs['zip_count']
            else:
                zip_sqft_med = self._nearest_zip_fallback('zip_med_sqft', px, py)
                zip_grade_med = self._nearest_zip_fallback('zip_med_grade', px, py)
                zip_age_med = self._nearest_zip_fallback('zip_med_age', px, py)
                zip_count = self._nearest_zip_fallback('zip_count', px, py)

            f['relative_sqft_to_zip_median'] = f['sqft_living'] / zip_sqft_med if zip_sqft_med else 1.0
            f['relative_grade_to_zip_median'] = f['grade'] / zip_grade_med if zip_grade_med else 1.0
            f['zip_density_indicator'] = float(zip_count)
            f['relative_age_to_zip_median'] = (f['age_at_sale'] / zip_age_med) if zip_age_med else 1.0

            bx = str(int(px // BLOCK_SIZE))
            by = str(int(py // BLOCK_SIZE))
            block_id = f"{bx}_{by}"
            if block_id in self.block_stats.index:
                f['oof_target_enc_block_price'] = float(self.block_stats.loc[block_id, 'block_mean_price'])
            else:
                f['oof_target_enc_block_price'] = self._nearest_block_fallback(px, py)

        return f

    def to_vector(self, f: dict) -> np.ndarray:
        return np.array([[f[name] for name in FEATURE_ORDER]], dtype=np.float64)
