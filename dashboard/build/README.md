# SpatialGAT Property Valuation Dashboard

A 5-tab Flask + vanilla JS dashboard built on top of the real King County
housing dataset (20,437 properties) and the real trained XGBoost baseline
model (`models/xgb_baseline_model.json`) from the Geospatial-Property-Valuation
repo, styled neon-on-black to match the reference dashboard screenshots.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

The app needs internet access in the *browser* (not the server) to load
Chart.js, Leaflet, and dark map tiles from CDNs — the Flask server itself
does not need internet access.

## What's real vs. what's computed live

- **Dataset**: the actual `kc_master_dataset_cleaned.parquet` (20,437 real
  King County property sales), copied from `dataset/processed/`.
- **Model**: the actual trained XGBoost booster (`xgb_baseline_model.json`),
  loaded and run for every prediction — not mocked.
- **Feature engineering** (`feature_engineering.py`): replicates the repo's
  spatial/temporal/ratio/target-encoding feature pipeline (`src/features/*.py`)
  so brand-new user-entered properties can be scored with the same 60
  features the model was trained on. Lat/long → the dataset's projected
  feet coordinates is done with a locally-fit affine transform (mean error
  ~30 ft vs. the original EPSG:2285 projection — negligible at city scale).
- **GNN metrics** (Tab 4): the real `gnn_eval_results.json`,
  `benchmark_comparison.csv`, `full_regression_audit.json`, and
  `cv_results.json` produced by the repo's own training/eval pipeline. The
  trained PyTorch Geometric GNN weights aren't re-loaded for live inference
  (no per-row GNN predictions were saved in the repo), so live predictions
  in Tabs 1–3 come from the real XGBoost baseline instead — the dashboard
  is upfront about this via the "XGBoost Baseline" labels next to GNN
  metrics wherever both appear.
- **k-NN attention map / attention-decay chart**: a genuine Gaussian RBF
  weighting computed from real property coordinates (same formula the repo
  uses to build graph edges), not pre-baked demo data.
- **Deal badges, spatial error heatmap, calibration scatter, similarity
  matcher**: all computed live from real model predictions and real
  feature vectors across the full dataset at startup.

## Project structure

```
app.py                     Flask backend, all API routes
feature_engineering.py     Live feature pipeline for new property inputs
data/                      Real dataset + real eval metrics (copied from repo)
models/                    Real trained XGBoost model (copied from repo)
templates/index.html       Single-page 5-tab layout
static/css/style.css       Neon-on-black theme
static/js/                 Tab logic (one file per tab) + shared helpers
```

## API endpoints

| Endpoint | Tab | Description |
|---|---|---|
| `GET /api/summary` | header | Top stat pills |
| `POST /api/valuate` | 1 | Point estimate, confidence range, structural/spatial breakdown, k-NN neighbors |
| `POST /api/deal-finder` | 2 | Filtered/ranked listing search with deal badges |
| `GET /api/similar/<id>` | 2 | Top-5 similar properties (feature-space nearest neighbors) |
| `POST /api/whatif` | 3 | Before/after valuation + ROI for a renovation scenario |
| `GET /api/model-performance` | 4 | Benchmark metrics, spatial heatmap, calibration, attention decay, CV folds |
| `GET /api/zipcodes` | 5 | List of zipcodes for the dropdown |
| `GET /api/neighborhood?zipcode=` | 5 | Local stats, distribution, top properties, local accuracy |
