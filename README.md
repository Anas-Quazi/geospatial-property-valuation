<div align="center">

# 🗺️ Geospatial Property Valuation

### Real Estate Pricing via Spatial Embeddings & Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10+-F0A500?style=flat-square&logo=python&logoColor=0D1117)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-F0A500?style=flat-square&logo=pytorch&logoColor=0D1117)
![PyG](https://img.shields.io/badge/PyG-torch--geometric-F0A500?style=flat-square&logo=pytorch&logoColor=0D1117)
![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial-F0A500?style=flat-square&logo=pandas&logoColor=0D1117)
![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-F0A500?style=flat-square&logo=xgboost&logoColor=0D1117)
![Flask](https://img.shields.io/badge/Flask-Dashboard-F0A500?style=flat-square&logo=flask&logoColor=0D1117)
![License](https://img.shields.io/badge/License-MIT-F0A500?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

</div>

---

```python
project = {
    "name"      : "Geospatial Property Valuation",
    "domain"    : "Construction & Real Estate",
    "approach"  : ["Spatial Embeddings", "KNN Graphs", "Graph Attention Networks"],
    "baseline"  : "XGBoost AVM",
    "dataset"   : "King County House Sales (USA)",
    "metric"    : "MAPE (Mean Absolute Percentage Error)",
    "internship": "Infotact Solutions — DSML Internship 2026",
    "authors"   : ["Shais013", "Anas-Quazi", "Aadi-1605", "Yash-Chattar"],
}
```

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Results](#results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Dashboard](#dashboard)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Git Workflow](#git-workflow)
- [Data Dictionary](#data-dictionary)
- [Team](#team)

---

## Problem Statement

Traditional Automated Valuation Models (AVMs) treat each property in isolation — relying only on bedrooms, square footage, and age. But a house's price is fundamentally **spatial**. It is shaped by its neighbors, nearby amenities, and localized socio-economic patterns that tabular models completely miss.

**This project proves it mathematically.**

We build a Graph Neural Network-based valuation engine that models every house as a node in a neighborhood graph, aggregating spatial context from its K-nearest neighbors to produce significantly more accurate price predictions than any standard AVM.

---

## Results

| Model | MAPE ↓ | RMSE ↓ | Notes |
|---|---|---|---|
| XGBoost Baseline (Spatial Block CV) | ~14.42% | ~$86K | 61 engineered features, tabular only |
| **SpatialGraphGAT** | **~7–9%** | **Lower** | 2-layer GAT with skip connection, log-price target |

> Evaluated on identical **spatial block cross-validation** folds to prevent geographic leakage. The GNN achieves relative MAPE improvement of **~38–50%** over the XGBoost baseline.

---

## Architecture

```
Raw Dataset  ──  lat/long + 21 tabular columns
        │
        ▼
┌─────────────────────────────────┐
│  Data Inspection                │  data_preprocessing/data_inspection.ipynb
│  + Preprocessing                │  data_preprocessing/data_preprocessing.ipynb
│                                 │  → sqft_basement fix, null fills, IQR outlier
│                                 │    removal, is_renovated flag, house_age feature
└──────────────┬──────────────────┘
               │  kc_house_cleaned.csv
               ▼
┌─────────────────────────────────┐
│  Feature Engineering            │  src/features/
│                                 │  ├── structural_ratio_features.py   (14 features)
│                                 │  ├── temporal_features.py
│                                 │  ├── Spatial_proximity_features.py
│                                 │  ├── oof_engine.py                  (OOF encodings)
│                                 │  └── master_merge.py
└──────────────┬──────────────────┘
               │  kc_master_dataset_cleaned.parquet  (~20K rows × 69 cols)
               ▼
┌─────────────────────────────────┐
│  XGBoost Baseline AVM           │  src/model_training/xgb_baseline.py
│                                 │  → n_estimators=500, lr=0.05, max_depth=6
│  Spatial Block CV               │  src/model_training/spatial_cv.py
│                                 │  → Mean CV MAPE ≈ 14.42%, RMSE ≈ $86K
│  Regression Audit               │  src/model_training/baseline_regression_audit.py
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Geospatial Processing          │  src/data/
│                                 │  ├── geospatial_conversion.py
│                                 │  └── spatial_eda.py
│  Spatial Topology               │  src/graph_construction/build_spatial_topology.py
│                                 │  → Projects to EPSG:2285 (WA State Plane, feet)
│  KNN Graph Builder              │  src/graph_construction/build_knn_graph.py
│                                 │  → K=10, house=node, proximity=edge
│  Spatial Embeddings             │  src/graph_construction/spatial_embeddings.py
└──────────────┬──────────────────┘
               │  knn_graph.npz + pyg_spatial_graph.pt
               ▼
┌─────────────────────────────────┐
│  SpatialGraphGAT                │  src/graph_construction/train_gnn.py
│                                 │  → 2-layer Graph Attention Network
│  Layer 1: GATConv(in, 128, h=8) │  → Multi-head attention over neighborhood
│  Layer 2: GATConv(1024, 128, h=4)│  → Aggregated spatial context
│  MLP Head + Skip Connection     │  → Raw features concatenated with graph output
│                                 │  → Huber loss, log1p price target, AdamW + LR scheduler
│  Benchmark Comparison           │  src/graph_construction/benchmark_comparison.py
└──────────────┬──────────────────┘
               │  gnn_eval_results.json + benchmark_comparison.csv
               ▼
┌─────────────────────────────────┐
│  Flask Dashboard                │  dashboard/build/
│                                 │  → 5-tab interactive analytics app
│                                 │  → Live inference, deal-finder, what-if,
│                                 │     regression audit, neighborhood explorer
└─────────────────────────────────┘
```

---

## Repository Structure

```
geospatial-property-valuation/
│
├── dataset/
│   ├── kc_house_data.csv                    # Raw King County dataset (21,613 rows)
│   ├── kc_house_cleaned.csv                 # Post-preprocessing cleaned dataset
│   └── processed/
│       ├── kc_master_dataset_cleaned.parquet  # Full feature-engineered master dataset
│       ├── kc_house_spatial.parquet           # Spatial columns added
│       ├── master_static_features.parquet     # Static feature subset
│       ├── knn_graph.npz                      # Sparse KNN adjacency matrix
│       ├── pyg_spatial_graph.pt               # PyG Data object (graph + features)
│       ├── pyg_spatial_graph_audited.pt       # Audited graph (post spatial_audit.py)
│       ├── spatial_validation_splits.parquet  # Spatial block CV fold assignments
│       ├── spatial_eda_heatmap.html           # Interactive Folium price heatmap
│       ├── cv_results.json                    # XGBoost spatial CV metrics
│       ├── full_regression_audit.json         # XGBoost residual analysis
│       ├── gnn_eval_results.json              # GNN test metrics (MAPE, RMSE, MAE)
│       └── benchmark_comparison.csv           # Side-by-side model comparison
│
├── data_preprocessing/
│   ├── data_inspection.ipynb                # Schema, null analysis, outlier flagging
│   └── data_preprocessing.ipynb            # Full cleaning pipeline (Shais013)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── geospatial_conversion.py         # lat/long → EPSG:2285 projection
│   │   ├── spatial_eda.py                   # EDA and Folium price heatmap
│   │   └── spatial_validation_splits.py     # Spatial block CV fold generator
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── structural_ratio_features.py     # 14 ratio features (bath_per_bed, etc.)
│   │   ├── temporal_features.py             # Age, renovation, cycle features
│   │   ├── Spatial_proximity_features.py    # Distance to city center, zip aggregates
│   │   ├── oof_engine.py                    # Out-of-fold target encoding
│   │   ├── clean_feature_dataset.py         # Post-merge cleanup
│   │   └── master_merge.py                  # Merge all feature groups
│   │
│   ├── graph_construction/
│   │   ├── build_spatial_topology.py        # Topology design + CRS validation
│   │   ├── build_knn_graph.py               # K=10 KNN graph (scipy/sklearn)
│   │   ├── spatial_embeddings.py            # Neighborhood embedding generation
│   │   ├── spatial_audit.py                 # Graph integrity checks
│   │   ├── train_gnn.py                     # SpatialGraphGAT training loop
│   │   └── benchmark_comparison.py          # XGBoost vs GNN comparison report
│   │
│   └── model_training/
│       ├── xgb_baseline.py                  # XGBoost training script (Shais013)
│       ├── spatial_cv.py                    # Spatial block cross-validation
│       └── baseline_regression_audit.py     # Residual analysis + SHAP (optional)
│
├── models/
│   └── xgb_baseline_model.json              # Saved XGBoost model weights
│
├── dashboard/
│   └── build/
│       ├── app.py                           # Flask server (5 REST endpoints)
│       ├── feature_engineering.py           # ReferenceContext + FEATURE_ORDER
│       ├── requirements.txt
│       ├── data/                            # Copied runtime artifacts
│       ├── models/                          # Copied model weights
│       ├── static/
│       │   ├── css/style.css
│       │   └── js/
│       │       ├── tab1_valuation.js
│       │       ├── tab2_dealfinder.js
│       │       ├── tab3_whatif.js
│       │       ├── tab4_audit.js
│       │       └── tab5_neighborhood.js
│       └── templates/index.html
│
├── docs/
│   └── data_dictionary.md                   # Column-level feature documentation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Walkthrough

### Phase 0 — Data Preprocessing

**Notebook:** `data_preprocessing/data_preprocessing.ipynb`

Cleans the raw King County dataset across six steps:

| Step | Action |
|---|---|
| `sqft_basement` | Converts `?` sentinel values to `0` and casts to float |
| Null fills | `waterfront` → 0, `view` → 0, `yr_renovated` → 0 |
| Feature engineering | `house_age = 2015 − yr_built`, `is_renovated` binary flag |
| Bedroom outliers | Removes properties with > 10 bedrooms |
| Price outliers | IQR-based removal (Q1 − 1.5×IQR to Q3 + 1.5×IQR) |
| Output | `dataset/kc_house_cleaned.csv` |

---

### Phase 1 — Feature Engineering

**Module:** `src/features/`

Four feature categories are engineered and merged into the master dataset:

**Structural Ratios** (`structural_ratio_features.py`) — 14 features including:
`bath_per_bed`, `sqft_per_bed`, `above_to_living_ratio`, `grade_condition_score`, `room_density`, `basement_ratio`, `living_to_lot_ratio`, and more.

**Temporal Features** (`temporal_features.py`) — `house_age`, `is_renovated`, `years_since_renovation`, `renovation_recency_score`.

**Spatial Proximity Features** (`Spatial_proximity_features.py`) — Distances to Seattle city center, zip-level price aggregates, spatial lag features.

**OOF Encodings** (`oof_engine.py`) — Out-of-fold target encodings for zip code and block, guarding against target leakage.

All features are merged via `master_merge.py` → `kc_master_dataset_cleaned.parquet` (~20,437 rows × 69 columns).

---

### Phase 2 — XGBoost Baseline

**Script:** `src/model_training/xgb_baseline.py`

Trains a spatially-aware XGBoost regressor on 61 features (leakage columns explicitly excluded via `IGNORE_COLS`).

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

**Spatial Block CV** (`src/model_training/spatial_cv.py`) evaluates the model on geographically non-overlapping folds:

```
Mean CV RMSE : ~$86,000
Mean CV MAPE : ~14.42%
```

This establishes the target to beat.

---

### Phase 3 — Graph Construction

**Module:** `src/graph_construction/`

| Script | Purpose |
|---|---|
| `build_spatial_topology.py` | Validates CRS, projects to EPSG:2285, defines K=10 topology |
| `build_knn_graph.py` | Constructs KNN graph; each house is a node, edges connect K=10 spatial neighbors |
| `spatial_embeddings.py` | Generates dense neighborhood embeddings from graph structure |
| `spatial_audit.py` | Integrity checks on node/edge counts, feature alignment |

**Coordinate system:** EPSG:2285 (Washington State Plane North, feet) — provides accurate Euclidean distances for neighborhood graph construction without spherical distortion.

**Graph statistics:**
```
Nodes  |V| : ~20,437 (one per property)
Edges  |E| : ~204,370 (K=10 directed edges per node)
```

---

### Phase 4 — Graph Attention Network

**Script:** `src/graph_construction/train_gnn.py`

**Model architecture:**

```python
class SpatialGraphGAT(nn.Module):
    # Layer 1: Multi-head attention — captures diverse neighborhood patterns
    conv1 = GATConv(in_channels, hidden=128, heads=8, concat=True)   # → 1024 dims
    # Layer 2: Aggregation
    conv2 = GATConv(1024, hidden=128, heads=4, concat=False)          # → 128 dims
    # MLP Head with skip connection (raw features concatenated)
    fc1   = Linear(128 + in_channels, 64)
    fc2   = Linear(64, 1)
```

**Training details:**
- **Target:** `log1p(price)` — stabilizes training and compresses price range
- **Loss:** Huber loss (robust to high-value outliers)
- **Optimizer:** Adam with `weight_decay=1e-4`
- **Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=10)
- **Early stopping:** patience=30 epochs
- **Feature scaling:** Standardized using train-split statistics only (no leakage)

**Key design decision — skip connection:** Raw house features are concatenated with the graph-aggregated neighborhood context before the MLP head. This ensures the model cannot regress purely on neighbor signals while losing individual property characteristics.

---

### Phase 5 — Benchmark & Dashboard

**Benchmark:** `src/graph_construction/benchmark_comparison.py` loads both `cv_results.json` (XGBoost) and `gnn_eval_results.json` (GNN), computes relative improvement per metric, and writes `benchmark_comparison.csv`.

---

## Dashboard

A 5-tab **Flask + Leaflet.js** analytics app (`dashboard/build/app.py`) that runs live inference using the trained XGBoost model against the full dataset.

| Tab | Feature |
|---|---|
| **Valuation** | Input property specs → get a price estimate with confidence interval |
| **Deal Finder** | Flags properties as Undervalued / Fair / Overpriced (±10% of model expectation) |
| **What-If** | Adjust features interactively and watch price update in real time |
| **Audit** | Residual analysis — where does the model over/under-predict and why |
| **Neighborhoods** | KNN-based similar property explorer with a spatial heatmap |

**Run the dashboard:**
```bash
cd dashboard/build
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | `pandas`, `numpy`, `scikit-learn`, `scipy` |
| Geospatial | `GeoPandas`, `Shapely`, `pyproj`, `Folium`, `rtree` |
| Graph Construction | `scipy.sparse`, `sklearn.neighbors.NearestNeighbors` |
| ML Baseline | `XGBoost` |
| Deep Learning | `PyTorch`, `torch-geometric` (GATConv) |
| Visualization | `Matplotlib`, `Seaborn`, `Folium`, `Leaflet.js`, `Chart.js` |
| Dashboard | `Flask`, vanilla JS |
| Coordinate System | EPSG:2285 — Washington State Plane North (feet) |
| Versioning | `Git`, GitHub Projects (Kanban) |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for GNN training (CPU works, slower)

### 1. Clone the repository

```bash
git clone https://github.com/Anas-Quazi/geospatial-property-valuation.git
cd geospatial-property-valuation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch note:** Install the correct PyTorch build for your hardware **before** running `pip install -r requirements.txt`. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 3. Run the pipeline (in order)

```bash
# Step 1 — Preprocessing (run notebook or export as script)
jupyter nbconvert --to script data_preprocessing/data_preprocessing.ipynb --execute

# Step 2 — Feature engineering + master merge
python src/features/master_merge.py

# Step 3 — XGBoost baseline training
python src/model_training/xgb_baseline.py

# Step 4 — Spatial CV evaluation
python src/model_training/spatial_cv.py

# Step 5 — Build KNN graph
python src/graph_construction/build_spatial_topology.py
python src/graph_construction/build_knn_graph.py
python src/graph_construction/spatial_embeddings.py

# Step 6 — Train GNN
python src/graph_construction/train_gnn.py

# Step 7 — Benchmark comparison
python src/graph_construction/benchmark_comparison.py
```

### 4. Launch the dashboard

```bash
cd dashboard/build
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Git Workflow

This project follows a **semantic commit + Kanban** approach as required by Infotact's evaluation protocol. All contributors work on individual branches and raise PRs to `main` at the end of each development week.

**Branch structure:**
```
main
├── dev/shais
├── dev/anas
├── dev/aadi
└── dev/yash
```

**Commit convention:**
```
feat: add structural ratio feature engineering (14 features)
model: train XGBoost baseline on master dataset (MAPE 14.42%)
graph: build KNN spatial topology with EPSG:2285 projection
fix: revert merge conflict on GeoDataFrame-based implementation
docs: add data_dictionary.md with column-level descriptions
```

**PR policy:** PRs must include a description of what changed and how it affects downstream pipeline steps. No force pushes to `main`; merge conflicts resolved via `git revert -m 1 HEAD` + clean PR.

> ⚠️ All notebooks committed with cleared outputs (`Kernel → Restart & Clear Output`). Model weights excluded via `.gitignore`.

---

## Data Dictionary

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for full column-level documentation covering the raw dataset, engineered features, and graph construction outputs.

**Key columns in `kc_master_dataset_cleaned.parquet`:**

| Column | Type | Description |
|---|---|---|
| `price` | float | Sale price (target variable) |
| `lat`, `long` | float | WGS84 coordinates |
| `x_proj`, `y_proj` | float | EPSG:2285 projected coordinates (feet) |
| `sqft_living` | float | Interior living area |
| `grade` | int | King County construction quality grade (1–13) |
| `house_age` | int | `2015 − yr_built` |
| `is_renovated` | int | Binary flag: `yr_renovated > 0` |
| `bath_per_bed` | float | Bathrooms per bedroom ratio |
| `above_to_living_ratio` | float | Above-ground to total living ratio |
| `grade_condition_score` | float | Product of grade and condition |
| `oof_target_enc_zip_price` | float | Out-of-fold zip-level price encoding |
| `fold` | int | Spatial block CV fold assignment |

---

## Team

| Name | GitHub | Primary Contributions |
|---|---|---|
| Shais | [Shais013](https://github.com/Shais013) | Data preprocessing, structural ratio features, XGBoost baseline, spatial topology, data dictionary |
| Anas | [Anas-Quazi](https://github.com/Anas-Quazi) | Project lead, repository structure, pipeline integration |
| Aadi | [Aadi-1605](https://github.com/Aadi-1605) | Data inspection, spatial block CV |
| Yash | [Yash-Chattar](https://github.com/Yash-Chattar) | Benchmark comparison, dashboard frontend |

---

<div align="center">

**Infotact Solutions — DSML Internship 2026 · Project 2 of 2**

</div>
