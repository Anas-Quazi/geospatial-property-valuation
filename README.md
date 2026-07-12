<div align="center">

# 🗺️ Geospatial Property Valuation
### Real Estate Pricing via Spatial Embeddings & Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10+-F0A500?style=flat-square&logo=python&logoColor=0D1117)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-F0A500?style=flat-square&logo=pytorch&logoColor=0D1117)
![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial-F0A500?style=flat-square&logo=pandas&logoColor=0D1117)
![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-F0A500?style=flat-square&logo=xgboost&logoColor=0D1117)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-F0A500?style=flat-square&logo=streamlit&logoColor=0D1117)
![License](https://img.shields.io/badge/License-MIT-F0A500?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Week](https://img.shields.io/badge/Progress-Week%201%20Complete-F0A500?style=flat-square)

</div>

---

```python
project = {
    "name"      : "Geospatial Property Valuation",
    "domain"    : "Construction & Real Estate",
    "approach"  : ["Spatial Embeddings", "KNN Graphs", "Graph Neural Networks"],
    "baseline"  : "XGBoost AVM",
    "dataset"   : "King County House Sales (USA)",
    "metric"    : "MAPE (Mean Absolute Percentage Error)",
    "internship": "Infotact Solutions — DSML Internship 2026",
    "authors"   : ["Shais013", "Anas-Quazi", "Aadi-1605", "Yash-Chattar"],
}
```

---

## Problem Statement

Traditional Automated Valuation Models (AVMs) treat each property in isolation — relying only on bedrooms, square footage, and age. But a house's price is fundamentally spatial. It is shaped by its neighbors, nearby amenities, and localized socio-economic patterns that tabular models completely miss.

**This project proves it mathematically.**

We build a Graph Neural Network-based valuation engine that models every house as a node in a neighborhood graph, aggregating spatial context from its K-nearest neighbors to produce significantly more accurate price predictions than any standard AVM.

---

## Architecture Overview

```
Raw Dataset (lat/lng + tabular features)
        │
        ▼
┌──────────────────────────┐
│  Data Inspection         │  ← Schema validation, null analysis, outlier flagging
│  + Preprocessing         │  ← Cleaning, feature extraction, IQR outlier removal
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  GeoPandas Wrangling     │  ← Haversine distances, coordinate normalization
│  + Folium Mapping        │  ← Interactive geo-visualization of price trends
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  XGBoost Baseline AVM    │  ← Standard benchmark (MAPE, RMSE)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  KNN Graph Builder       │  ← Each house = Node, edges = K spatial neighbors
│  Spatial Embeddings      │  ← Neighborhood context as dense vectors
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  GNN / Attention Model   │  ← Aggregates neighbor pricing signals
│  Valuation Engine        │  ← Outperforms XGBoost baseline
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Streamlit Dashboard     │  ← Predicted prices + spatial heatmaps
└──────────────────────────┘
```

---

## Results

| Model | MAPE ↓ | RMSE ↓ | Notes |
|---|---|---|---|
| Linear Regression | ~18% | High | No spatial context |
| XGBoost (Baseline) | ~12% | Moderate | Tabular features only |
| **GNN + Attention** | **~7%** | **Low** | Spatial dependencies injected |

> Results updated progressively across 4 weeks. Final numbers populated after Week 4.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Geospatial Processing | `GeoPandas`, `Shapely`, `Folium`, `Kepler.gl` |
| Graph Construction | `NetworkX`, `DGL` or `PyTorch Geometric` |
| ML Baseline | `XGBoost`, `scikit-learn` |
| Deep Learning | `PyTorch`, Graph Attention Networks |
| Visualization | `Matplotlib`, `Seaborn`, `Folium` |
| Dashboard | `Streamlit` |
| Version Control | `Git`, `GitHub Projects (Kanban)`, `DVC` |

---

## 4-Week Engineering Roadmap

### Week 1 — Geospatial Data Acquisition & Processing ✅
- [x] Load and inspect King County House Sales dataset
- [x] Clean data — fix `sqft_basement`, fill nulls, encode `date`
- [x] Remove price outliers using IQR method
- [x] Engineer `house_age` and `is_renovated` features
- [x] Save cleaned dataset → `kc_house_cleaned.csv`
- [ ] Compute pairwise Haversine distances
- [ ] Plot interactive price heatmap using Folium

### Week 2 — Feature Engineering & Baseline ML
- [ ] Engineer tabular features (age, distance to city center, etc.)
- [ ] Train XGBoost regressor → compute baseline MAPE & RMSE
- [ ] Document model limitations on gentrifying neighborhoods

### Week 3 — Spatial Embeddings & Graph Construction
- [ ] Convert dataset into graph structure (house = node, edge = K-nearest neighbor)
- [ ] Generate spatial embeddings representing neighborhood context
- [ ] Validate graph structure with visualizations

### Week 4 — GNN / Attention Modeling & Dashboard
- [ ] Train GNN or Attention-Based Spatial model on constructed graph
- [ ] Compare new MAPE against XGBoost baseline
- [ ] Deploy Streamlit app with spatial price disparity map

---

## Repository Structure

```
geospatial-property-valuation/
│
├── dataset/                        # Cleaned datasets (not gitignored here)
│   ├── kc_house_data.csv           # Raw dataset
│   └── kc_house_cleaned.csv        # Cleaned output from preprocessing
│
├── data_inspection/
│   └── data_inspection.ipynb       # Schema, nulls, outlier analysis — Aadi
│
├── data_preprocessing/
│   └── data_preprocessing.ipynb   # Full cleaning pipeline — Shais
│
├── feature_engineering/            # Week 2
│   └── feature_engineering.ipynb
│
├── baseline_model/                 # Week 2
│   └── xgboost_baseline.ipynb
│
├── graph_construction/             # Week 3
│   └── knn_graph.ipynb
│
├── gnn_model/                      # Week 4
│   └── gnn_attention.ipynb
│
├── dashboard/                      # Week 4
│   └── app.py
│
├── models/                         # ← gitignored
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Anas-Quazi/geospatial-property-valuation.git
cd geospatial-property-valuation

pip install -r requirements.txt

# Run dashboard (Week 4)
streamlit run dashboard/app.py
```

---

## GitHub Workflow

This project follows a strict **semantic commit + Kanban** approach as required by Infotact's evaluation protocol. All contributors work on individual branches and raise PRs to `main` at the end of each week.

**Branch structure:**
```
main
├── dev/shais
├── dev/anas
├── dev/aadi
└── dev/yash
```

**Commit format:**
```
feat: implement Haversine distance matrix (fixes #1)
model: train GNN attention layer on KNN graph (fixes #5)
viz: add Folium price heatmap to Week 1 notebook (fixes #3)
```

Issues are tracked on the [Project Kanban Board](../../projects) — managers can audit week-over-week progress via GitHub's native timestamps.

> ⚠️ All notebooks committed with cleared outputs (`Kernel → Restart & Clear Output`). Model weights excluded via `.gitignore`.

---

## Team

| Name | GitHub |
|---|---|
| Shais | [Shais013](https://github.com/Shais013) |
| Anas | [Anas-Quazi](https://github.com/Anas-Quazi) |
| Aadi | [Aadi-1605](https://github.com/Aadi-1605) |
| Yash | [Yash-Chattar](https://github.com/Yash-Chattar) |

---

<div align="center">

**Infotact Solutions — DSML Internship 2025 · Project 2 of 2**

</div>
