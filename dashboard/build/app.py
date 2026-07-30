import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from feature_engineering import ReferenceContext, FEATURE_ORDER

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load real data + real trained model once at startup
# ---------------------------------------------------------------------------
print("Loading dataset...")
DF = pd.read_parquet(os.path.join(DATA_DIR, "kc_master_dataset_cleaned.parquet")).reset_index(drop=True)

print("Loading XGBoost model...")
BOOSTER = xgb.Booster()
BOOSTER.load_model(os.path.join(MODEL_DIR, "xgb_baseline_model.json"))

print("Building reference context (zip/block aggregates, affine projection)...")
REF = ReferenceContext(DF)

with open(os.path.join(DATA_DIR, "gnn_eval_results.json")) as fh:
    GNN_EVAL = json.load(fh)
with open(os.path.join(DATA_DIR, "full_regression_audit.json")) as fh:
    XGB_AUDIT = json.load(fh)
with open(os.path.join(DATA_DIR, "cv_results.json")) as fh:
    CV_RESULTS = json.load(fh)
BENCHMARK = pd.read_csv(os.path.join(DATA_DIR, "benchmark_comparison.csv"))

MEAN_MAE = XGB_AUDIT["mean_mae"]
MEAN_MAPE = XGB_AUDIT["mean_mape"]


def predict_batch(feature_dicts):
    X = np.array([[f[name] for name in FEATURE_ORDER] for f in feature_dicts], dtype=np.float64)
    dm = xgb.DMatrix(X, feature_names=FEATURE_ORDER)
    return BOOSTER.predict(dm)


def predict_one(feature_dict):
    v = REF.to_vector(feature_dict)
    dm = xgb.DMatrix(v, feature_names=FEATURE_ORDER)
    return float(BOOSTER.predict(dm)[0])


print("Running real model inference across full dataset (one-time, for audit/deal-finder/neighborhoods)...")
X_ALL = DF[FEATURE_ORDER].to_numpy(dtype=np.float64)
DM_ALL = xgb.DMatrix(X_ALL, feature_names=FEATURE_ORDER)
DF["predicted_price"] = BOOSTER.predict(DM_ALL)
DF["residual"] = DF["price"] - DF["predicted_price"]
DF["abs_pct_error"] = (DF["residual"].abs() / DF["price"]) * 100

# Deal badge: actual vs model's "expected" price for that archetype
DEAL_LOW = 0.90
DEAL_HIGH = 1.10


def deal_badge(row):
    ratio = row["price"] / row["predicted_price"]
    if ratio <= DEAL_LOW:
        return "Undervalued"
    elif ratio >= DEAL_HIGH:
        return "Overpriced"
    return "Fair"


DF["deal_badge"] = DF.apply(deal_badge, axis=1)

# ---------------------------------------------------------------------------
# Similarity / latent-space matcher: standardized feature vector + kNN
# ---------------------------------------------------------------------------
SIM_COLS = [
    "sqft_living", "bedrooms", "bathrooms", "grade", "condition",
    "lat", "long", "house_age", "view", "waterfront", "price_per_sqft",
]
SCALER = StandardScaler()
SIM_MATRIX = SCALER.fit_transform(DF[SIM_COLS].to_numpy(dtype=np.float64))
NN_MODEL = NearestNeighbors(n_neighbors=6, algorithm="auto").fit(SIM_MATRIX)

print(f"Ready. {len(DF)} real properties loaded, model MAPE={GNN_EVAL['mape']:.2f}% (GNN) vs "
      f"{BENCHMARK.loc[BENCHMARK.Metric=='MAPE', 'XGBoost Baseline'].values[0]:.2f}% (XGBoost baseline).")


# ---------------------------------------------------------------------------
# Routes - pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API - global summary (top stat pills)
# ---------------------------------------------------------------------------
@app.route("/api/summary")
def api_summary():
    return jsonify({
        "total_properties": int(len(DF)),
        "avg_price": float(DF["price"].mean()),
        "median_price": float(DF["price"].median()),
        "zipcodes": int(DF["zipcode"].nunique()),
        "gnn_mape": GNN_EVAL["mape"],
        "xgb_mape": float(BENCHMARK.loc[BENCHMARK.Metric == "MAPE", "XGBoost Baseline"].values[0]),
        "improvement_pct": float(BENCHMARK.loc[BENCHMARK.Metric == "MAPE", "Improvement (%)"].values[0]),
    })


# ---------------------------------------------------------------------------
# TAB 1 - Property Valuation Engine
# ---------------------------------------------------------------------------
@app.route("/api/valuate", methods=["POST"])
def api_valuate():
    raw = request.get_json(force=True)
    required = ["lat", "long", "sqft_living", "sqft_lot", "bedrooms", "bathrooms",
                "condition", "grade", "yr_built"]
    missing = [k for k in required if k not in raw]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    feats = REF.build_features(raw)
    predicted = predict_one(feats)

    neutral_feats = REF.build_features(raw, neutralize_location=True)
    structural_baseline = predict_one(neutral_feats)
    spatial_contribution = predicted - structural_baseline

    half_range = max(MEAN_MAE, predicted * (MEAN_MAPE / 100.0))
    low = max(0, predicted - half_range)
    high = predicted + half_range

    # k-NN attention map: nearest real properties by projected (feet) distance
    px, py = feats["projected_x"], feats["projected_y"]
    dx = DF["projected_x"].to_numpy() - px
    dy = DF["projected_y"].to_numpy() - py
    dist = np.sqrt(dx ** 2 + dy ** 2)
    k = 8
    idx = np.argsort(dist)[:k]
    nearest_dist = dist[idx]
    sigma = max(nearest_dist.mean(), 1.0)
    weights = np.exp(-(nearest_dist ** 2) / (2 * sigma ** 2))
    weights = weights / weights.sum()

    neighbors = []
    for rank, (i, d, w) in enumerate(zip(idx, nearest_dist, weights)):
        row = DF.iloc[i]
        neighbors.append({
            "id": int(row["id"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "price": float(row["price"]),
            "sqft_living": int(row["sqft_living"]),
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "grade": int(row["grade"]),
            "condition": int(row["condition"]),
            "yr_built": int(row["yr_built"]),
            "distance_ft": float(d),
            "distance_mi": float(d / 5280.0),
            "attention_weight": float(w),
        })

    # Property vs neighborhood (zip) comparison — real zip-level aggregates
    zipcode = int(feats["zipcode"])
    subject_psqft = predicted / raw["sqft_living"] if raw["sqft_living"] else None
    zip_comparison = None
    if zipcode in REF.zip_stats.index:
        zs = REF.zip_stats.loc[zipcode]
        zip_comparison = {
            "zipcode": zipcode,
            "count": int(zs["zip_count"]),
            "subject": {
                "sqft_living": raw["sqft_living"], "grade": raw["grade"],
                "price_per_sqft": subject_psqft,
            },
            "neighborhood_avg": {
                "sqft_living": float(zs["zip_med_sqft"]), "grade": float(zs["zip_med_grade"]),
                "price_per_sqft": float(zs["zip_mean_psqft"]),
            },
        }

    return jsonify({
        "predicted_price": predicted,
        "confidence_low": low,
        "confidence_high": high,
        "structural_value": structural_baseline,
        "spatial_value": spatial_contribution,
        "resolved_zipcode": zipcode,
        "neighbors": neighbors,
        "zip_comparison": zip_comparison,
    })


# ---------------------------------------------------------------------------
# TAB 2 - Deal Finder & Feature Matcher
# ---------------------------------------------------------------------------
@app.route("/api/deal-finder", methods=["POST"])
def api_deal_finder():
    body = request.get_json(force=True) or {}
    sub = DF
    if body.get("min_sqft") is not None:
        sub = sub[sub["sqft_living"] >= body["min_sqft"]]
    if body.get("max_sqft") is not None:
        sub = sub[sub["sqft_living"] <= body["max_sqft"]]
    if body.get("min_beds") is not None:
        sub = sub[sub["bedrooms"] >= body["min_beds"]]
    if body.get("max_beds") is not None:
        sub = sub[sub["bedrooms"] <= body["max_beds"]]
    if body.get("min_baths") is not None:
        sub = sub[sub["bathrooms"] >= body["min_baths"]]
    if body.get("max_baths") is not None:
        sub = sub[sub["bathrooms"] <= body["max_baths"]]
    if body.get("min_price") is not None:
        sub = sub[sub["price"] >= body["min_price"]]
    if body.get("max_price") is not None:
        sub = sub[sub["price"] <= body["max_price"]]
    if body.get("zipcode"):
        sub = sub[sub["zipcode"] == int(body["zipcode"])]

    expected_market_price = float(sub["predicted_price"].mean()) if len(sub) else None
    sub = sub.sort_values("residual", ascending=True)  # most-undervalued first

    limit = int(body.get("limit", 50))
    out = []
    for _, row in sub.head(limit).iterrows():
        out.append({
            "id": int(row["id"]),
            "price": float(row["price"]),
            "predicted_price": float(row["predicted_price"]),
            "sqft_living": int(row["sqft_living"]),
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "zipcode": int(row["zipcode"]),
            "grade": int(row["grade"]),
            "condition": int(row["condition"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "deal_badge": row["deal_badge"],
            "diff_pct": float((row["price"] / row["predicted_price"] - 1) * 100),
        })

    return jsonify({
        "count": int(len(sub)),
        "expected_market_price": expected_market_price,
        "results": out,
    })


@app.route("/api/similar/<int:property_id>")
def api_similar(property_id):
    matches = DF.index[DF["id"] == property_id].tolist()
    if not matches:
        return jsonify({"error": "property not found"}), 404
    i = matches[0]
    vec = SIM_MATRIX[i].reshape(1, -1)
    dist, idx = NN_MODEL.kneighbors(vec, n_neighbors=6)
    out = []
    for d, j in zip(dist[0], idx[0]):
        if j == i:
            continue
        row = DF.iloc[j]
        out.append({
            "id": int(row["id"]),
            "price": float(row["price"]),
            "sqft_living": int(row["sqft_living"]),
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "grade": int(row["grade"]),
            "zipcode": int(row["zipcode"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "similarity": float(1.0 / (1.0 + d)),
        })
    out = out[:5]
    base = DF.iloc[i]
    return jsonify({
        "base": {
            "id": int(base["id"]), "price": float(base["price"]),
            "sqft_living": int(base["sqft_living"]), "bedrooms": int(base["bedrooms"]),
            "bathrooms": float(base["bathrooms"]), "grade": int(base["grade"]),
            "zipcode": int(base["zipcode"]),
        },
        "similar": out,
    })


# ---------------------------------------------------------------------------
# TAB 3 - Renovation / What-If Simulator
# ---------------------------------------------------------------------------
@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    body = request.get_json(force=True)
    base_raw = body["base"]
    delta = body.get("delta", {})

    before_feats = REF.build_features(base_raw)
    before_price = predict_one(before_feats)

    after_raw = dict(base_raw)
    after_raw["sqft_living"] = base_raw["sqft_living"] + delta.get("add_sqft", 0)
    after_raw["bedrooms"] = base_raw["bedrooms"] + delta.get("add_beds", 0)
    after_raw["bathrooms"] = base_raw["bathrooms"] + delta.get("add_baths", 0)
    after_raw["grade"] = min(13, base_raw["grade"] + delta.get("grade_upgrade", 0))
    after_raw["condition"] = min(5, base_raw["condition"] + delta.get("condition_upgrade", 0))
    if delta.get("add_sqft", 0) > 0:
        after_raw["sqft_above"] = base_raw.get("sqft_above", base_raw["sqft_living"]) + delta.get("add_sqft", 0)

    after_feats = REF.build_features(after_raw)
    after_price = predict_one(after_feats)

    # Cumulative step-by-step contribution (waterfall): apply each change one
    # at a time, in order, over the real model — so the chart shows exactly
    # how much of the total gain each individual upgrade is responsible for.
    steps = [("Base", {})]
    if delta.get("add_sqft", 0):
        steps.append((f"+{int(delta['add_sqft'])} sqft", {"sqft_living": delta.get("add_sqft", 0)}))
    if delta.get("add_beds", 0):
        steps.append((f"+{delta['add_beds']:g} bed", {"bedrooms": delta.get("add_beds", 0)}))
    if delta.get("add_baths", 0):
        steps.append((f"+{delta['add_baths']:g} bath", {"bathrooms": delta.get("add_baths", 0)}))
    if delta.get("grade_upgrade", 0):
        steps.append((f"Grade +{int(delta['grade_upgrade'])}", {"grade": delta.get("grade_upgrade", 0)}))
    if delta.get("condition_upgrade", 0):
        steps.append((f"Condition +{int(delta['condition_upgrade'])}", {"condition": delta.get("condition_upgrade", 0)}))

    waterfall = [{"label": "Base", "price": before_price}]
    running = dict(base_raw)
    for label, change in steps[1:]:
        for k, v in change.items():
            if k == "grade":
                running[k] = min(13, running.get(k, base_raw.get("grade", 7)) + v)
            elif k == "condition":
                running[k] = min(5, running.get(k, base_raw.get("condition", 3)) + v)
            else:
                running[k] = running.get(k, 0) + v
        if "sqft_living" in change:
            running["sqft_above"] = base_raw.get("sqft_above", base_raw["sqft_living"]) + \
                (running["sqft_living"] - base_raw["sqft_living"])
        step_feats = REF.build_features(running)
        step_price = predict_one(step_feats)
        waterfall.append({"label": label, "price": step_price})

    reno_cost = body.get("renovation_cost")
    value_gain = after_price - before_price
    roi_pct = None
    if reno_cost and reno_cost > 0:
        roi_pct = ((value_gain - reno_cost) / reno_cost) * 100

    return jsonify({
        "before_price": before_price,
        "after_price": after_price,
        "value_gain": value_gain,
        "value_gain_pct": (value_gain / before_price) * 100 if before_price else None,
        "renovation_cost": reno_cost,
        "roi_pct": roi_pct,
        "waterfall": waterfall,
    })


# ---------------------------------------------------------------------------
# TAB 4 - Model Performance & Spatial Audit
# ---------------------------------------------------------------------------
@app.route("/api/model-performance")
def api_model_performance():
    # Spatial error heatmap: bin real residuals onto a lat/long grid
    n_bins = 22
    lat_bins = np.linspace(DF["lat"].min(), DF["lat"].max(), n_bins + 1)
    lon_bins = np.linspace(DF["long"].min(), DF["long"].max(), n_bins + 1)
    DF["_lat_bin"] = pd.cut(DF["lat"], lat_bins, labels=False, include_lowest=True)
    DF["_lon_bin"] = pd.cut(DF["long"], lon_bins, labels=False, include_lowest=True)
    grid = DF.groupby(["_lat_bin", "_lon_bin"]).agg(
        mean_abs_pct_error=("abs_pct_error", "mean"),
        count=("price", "size"),
        lat=("lat", "mean"),
        long=("long", "mean"),
    ).reset_index()
    grid = grid[grid["count"] >= 3]
    heatmap = [{
        "lat": float(r.lat), "long": float(r.long),
        "mape": float(r.mean_abs_pct_error), "count": int(r.count),
    } for r in grid.itertuples()]

    # Calibration scatter: sample real actual-vs-predicted pairs
    sample = DF.sample(min(600, len(DF)), random_state=42)
    calibration = [{"actual": float(a), "predicted": float(p)}
                   for a, p in zip(sample["price"], sample["predicted_price"])]

    # Attention decay vs distance: genuine Gaussian RBF weight curve, sigma
    # calibrated from the median real nearest-neighbor spacing in the graph.
    sample_pts = DF.sample(min(300, len(DF)), random_state=7)[["projected_x", "projected_y"]].to_numpy()
    from scipy.spatial import cKDTree
    tree = cKDTree(DF[["projected_x", "projected_y"]].to_numpy())
    d, _ = tree.query(sample_pts, k=9)
    median_nn_dist = float(np.median(d[:, 1:]))
    sigma = median_nn_dist * 1.5
    distances_ft = np.linspace(0, sigma * 4, 40)
    decay = np.exp(-(distances_ft ** 2) / (2 * sigma ** 2))
    attention_decay = [{"distance_mi": float(dft / 5280.0), "attention_weight": float(w)}
                        for dft, w in zip(distances_ft, decay)]

    # Real feature importance straight from the trained XGBoost booster (gain)
    importance = BOOSTER.get_score(importance_type="gain")
    top_importance = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:12]
    total_gain = sum(v for _, v in importance.items()) or 1.0
    feature_importance = [{"feature": k, "importance_pct": float(v / total_gain * 100)}
                           for k, v in top_importance]

    # Error distribution: real abs % error histogram across the full dataset
    err_bins = np.linspace(0, min(DF["abs_pct_error"].quantile(0.99), 60), 16)
    err_hist, err_edges = np.histogram(DF["abs_pct_error"].clip(upper=err_bins[-1]), bins=err_bins)
    error_distribution = [{"bin_start": float(err_edges[i]), "bin_end": float(err_edges[i + 1]),
                            "count": int(err_hist[i])} for i in range(len(err_hist))]

    return jsonify({
        "benchmark": BENCHMARK.to_dict(orient="records"),
        "gnn_eval": GNN_EVAL,
        "xgb_audit": XGB_AUDIT,
        "cv_results": CV_RESULTS,
        "spatial_heatmap": heatmap,
        "calibration": calibration,
        "attention_decay": attention_decay,
        "feature_importance": feature_importance,
        "error_distribution": error_distribution,
    })


# ---------------------------------------------------------------------------
# TAB 5 - Neighborhood Explorer
# ---------------------------------------------------------------------------
@app.route("/api/zipcodes")
def api_zipcodes():
    zips = sorted(DF["zipcode"].unique().tolist())
    return jsonify({"zipcodes": [int(z) for z in zips]})


@app.route("/api/neighborhood")
def api_neighborhood():
    zipcode = request.args.get("zipcode", type=int)
    if zipcode is None:
        return jsonify({"error": "zipcode required"}), 400
    sub = DF[DF["zipcode"] == zipcode]
    if len(sub) == 0:
        return jsonify({"error": "no properties found for this zipcode"}), 404

    price_bins = np.linspace(sub["price"].min(), sub["price"].max(), 12)
    hist, edges = np.histogram(sub["price"], bins=price_bins)
    distribution = [{"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(hist[i])}
                     for i in range(len(hist))]

    grade_mix = sub.groupby("grade").size().reset_index(name="count").sort_values("grade")
    grade_breakdown = [{"grade": int(r.grade), "count": int(r.count)} for r in grade_mix.itertuples()]

    bed_mix = sub.groupby("bedrooms").size().reset_index(name="count").sort_values("bedrooms")
    bed_breakdown = [{"bedrooms": int(r.bedrooms), "count": int(r.count)} for r in bed_mix.itertuples()]

    badge_counts = sub["deal_badge"].value_counts().to_dict()

    top = sub.sort_values("price", ascending=False).head(10)
    top_properties = [{
        "id": int(r.id), "price": float(r.price), "sqft_living": int(r.sqft_living),
        "bedrooms": int(r.bedrooms), "bathrooms": float(r.bathrooms),
        "lat": float(r.lat), "long": float(r.long), "deal_badge": r.deal_badge,
    } for r in top.itertuples()]

    return jsonify({
        "zipcode": zipcode,
        "count": int(len(sub)),
        "avg_price": float(sub["price"].mean()),
        "median_price": float(sub["price"].median()),
        "avg_price_per_sqft": float((sub["price"] / sub["sqft_living"]).mean()),
        "price_min": float(sub["price"].min()),
        "price_max": float(sub["price"].max()),
        "local_mape": float(sub["abs_pct_error"].mean()),
        "distribution": distribution,
        "top_properties": top_properties,
        "grade_breakdown": grade_breakdown,
        "bed_breakdown": bed_breakdown,
        "badge_counts": {
            "Undervalued": int(badge_counts.get("Undervalued", 0)),
            "Fair": int(badge_counts.get("Fair", 0)),
            "Overpriced": int(badge_counts.get("Overpriced", 0)),
        },
        "center_lat": float(sub["lat"].mean()),
        "center_long": float(sub["long"].mean()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
