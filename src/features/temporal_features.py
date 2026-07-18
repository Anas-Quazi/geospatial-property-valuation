import pandas as pd
import numpy as np
from pathlib import Path


def add_temporal_decay_features(df):
    """
    Adds temporal/decay features to the cleaned KC housing df.
    Assumes df already has: sale_year, sale_month, yr_built, yr_renovated,
    is_renovated, house_age (from data_preprocessing.ipynb).
    Does not recompute sale_year, sale_month, is_renovated, or house_age.
    """
    df = df.copy()

    # years_since_renovated (defaults to house_age if never renovated)
    df['years_since_renovated'] = np.where(
        df['is_renovated'] == 1,
        df['sale_year'] - df['yr_renovated'],
        df['house_age']
    )

    # yrs_between_built_and_renovated
    df['yrs_between_built_and_renovated'] = np.where(
        df['is_renovated'] == 1,
        df['yr_renovated'] - df['yr_built'],
        0
    )

    # material_decay_rate
    df['material_decay_rate'] = np.exp(-0.05 * df['house_age'])

    # renovated_decay_rate
    df['renovated_decay_rate'] = np.exp(-0.05 * df['years_since_renovated'])

    return df


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent
    input_path = BASE_DIR / "dataset" / "kc_house_cleaned.csv"

    df = pd.read_csv(input_path)
    df = add_temporal_decay_features(df)
    print(df[['house_age', 'is_renovated', 'years_since_renovated',
              'yrs_between_built_and_renovated', 'material_decay_rate',
              'renovated_decay_rate']].head())
