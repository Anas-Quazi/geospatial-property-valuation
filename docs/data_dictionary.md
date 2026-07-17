# Data Dictionary — King County House Sales Dataset

> Reference document for all features used in the Geospatial Property Valuation pipeline.

---

## Target Variable

| Column | Type | Description |
|--------|------|-------------|
| `price` | float | Sale price of the house (USD). Target variable for regression. |

---

## Identifier & Date

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique identifier for each house listing. Dropped before modeling. |
| `date` | object | Date the house was sold. Used for temporal context; not a direct feature. |

---

## Structural Features

| Column | Type | Description |
|--------|------|-------------|
| `bedrooms` | int | Number of bedrooms. Outlier at 33 removed during preprocessing. |
| `bathrooms` | float | Number of bathrooms (fractional values indicate half-baths). |
| `sqft_living` | int | Interior living area in square feet. |
| `sqft_lot` | int | Total lot area in square feet. |
| `floors` | float | Number of floors in the house. |
| `sqft_above` | int | Square footage above ground level. |
| `sqft_basement` | int | Square footage of basement. Parsed from mixed string/numeric format during preprocessing. |
| `sqft_living15` | int | Average interior living space of the 15 nearest neighbors (sq ft). |
| `sqft_lot15` | int | Average lot size of the 15 nearest neighbors (sq ft). |

---

## Condition & Quality

| Column | Type | Description |
|--------|------|-------------|
| `condition` | int | Overall condition rating (1–5). 1 = Poor, 5 = Excellent. |
| `grade` | int | Construction and design quality grade (1–13). Higher = better build quality. |
| `view` | int | Number of times the property has been viewed (0–4). Nulls filled with 0. |
| `waterfront` | int | Binary flag — 1 if the property has a waterfront view. Nulls filled with 0. |

---

## Location Features

| Column | Type | Description |
|--------|------|-------------|
| `lat` | float | Latitude coordinate of the property. Used for spatial embedding. |
| `long` | float | Longitude coordinate of the property. Used for spatial embedding. |
| `zipcode` | int | ZIP code of the property. Used for Zipcode GroupKFold CV strategy. |

---

## Temporal Features

| Column | Type | Description |
|--------|------|-------------|
| `yr_built` | int | Year the house was originally built. |
| `yr_renovated` | int | Year of last renovation. 0 indicates no renovation. Nulls filled with 0. |

---

## Engineered Features

| Column | Type | Description |
|--------|------|-------------|
| `house_age` | int | Age of the house at time of sale. Derived as `sale_year - yr_built`. |
| `is_renovated` | int | Binary flag — 1 if `yr_renovated > 0`, else 0. Encodes renovation status cleanly. |

---

## Dropped / Excluded Columns

| Column | Reason |
|--------|--------|
| `id` | Non-informative identifier. |
| `date` | Raw date string; not used as a direct model feature. |

---

## Preprocessing Notes

- **`sqft_basement`**: Originally stored as mixed string (`"?"`) and numeric values — parsed and cast to int.
- **`waterfront`, `view`**: Contained nulls — filled with `0` (absence assumed).
- **`yr_renovated`**: Nulls filled with `0`; binary `is_renovated` flag derived from it.
- **Outlier removal**: Record with `bedrooms == 33` dropped as a data entry error. IQR-based filtering applied on `price`.
