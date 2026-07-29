import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data

from build_spatial_topology import inspect_topology_data
from build_knn_graph import build_knn_graph

# ---------------------------------------------------------------------------
# Phase 3: Spatial Context Embedding Engineering
# ---------------------------------------------------------------------------
# Engineers local spatial lags, density metrics, and builds the PyG Data object.
# Owner : Yash-Chattar, Aadi-1605
# ---------------------------------------------------------------------------

# Columns excluded from the GNN feature matrix.
# Must stay in sync with IGNORE_COLS in src/model_training/xgb_baseline.py.
# projected_x/y and x_proj/y_proj are excluded because x_coords/y_coords
# (Cat 2) carry the same values — keeping duplicates inflates the feature
# matrix and biases spatial-lag computation.
IGNORE_COLS = {
    "price",          # target
    "price_per_sqft", # leakage: derived from price
    "geometry",       # non-numeric object column
    "block_id",       # spatial block identifier, not a feature
    "id",             # row identifier
    "fold",           # CV split assignment
    "projected_x",    # duplicate of x_coords (Cat 2)
    "projected_y",    # duplicate of y_coords (Cat 2)
    "x_proj",         # Phase 1 in-memory alias — same values as x_coords
    "y_proj",         # Phase 1 in-memory alias — same values as y_coords
}

TARGET_COL = "price"


def compute_spatial_lags(
    x_matrix: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Weighted neighbor average for each node feature.

    Uses the Gaussian decay edge weights from Phase 2, so closer neighbors
    contribute more to each node's spatial context embedding.
    """
    src, dst = edge_index
    weights = edge_attr.squeeze()

    spatial_lags = np.zeros_like(x_matrix)
    np.add.at(spatial_lags, src, x_matrix[dst] * weights[:, None])

    weight_sums = np.zeros(n_nodes)
    np.add.at(weight_sums, src, weights)
    weight_sums = np.maximum(weight_sums, 1e-8)  # avoid div-by-zero

    return spatial_lags / weight_sums[:, None]


def create_pyg_data(
    df: pd.DataFrame,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    feature_cols: list,
    coord_cols: tuple = ("x_proj", "y_proj"),
    target_col: str = TARGET_COL,
) -> Data:
    """Assembles node features, spatial lag embeddings, coordinates, and targets
    into a PyTorch Geometric Data instance.

    data.x   : [n_nodes, 2 * len(feature_cols)] — raw features + spatial lags
    data.pos : [n_nodes, 2] — EPSG:2285 projected coords for spatial audit
    data.y   : [n_nodes, 1] — raw price
    """
    n_nodes = len(df)

    # 1. Raw node features (all 60 engineered features)
    x_raw = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    # 2. Spatial lag embeddings (weighted neighbor averages)
    spatial_lags = compute_spatial_lags(x_raw, edge_index, edge_attr, n_nodes)

    # 3. Combine: base features || spatial lags
    x_combined = np.hstack([x_raw, spatial_lags]).astype(np.float32)

    # 4. Projected coordinates — stored separately for spatial block audit.
    #    Not included in x to avoid leaking positional signal directly into
    #    the feature matrix (spatial context is already captured via lags).
    pos = df[list(coord_cols)].to_numpy(dtype=np.float32)

    y = df[target_col].to_numpy(dtype=np.float32)

    data = Data(
        x=torch.from_numpy(x_combined),
        y=torch.from_numpy(y).unsqueeze(1),
        edge_index=torch.from_numpy(edge_index).to(torch.long),
        edge_attr=torch.from_numpy(edge_attr).to(torch.float32),
        pos=torch.from_numpy(pos),
        num_nodes=n_nodes,
    )

    return data


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    candidate_paths = [
        script_dir / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet",
        script_dir.parent / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet",
        script_dir.parent.parent / "dataset" / "processed" / "kc_master_dataset_cleaned.parquet",
        Path("dataset/processed/kc_master_dataset_cleaned.parquet"),
    ]
    input_path = next((p for p in candidate_paths if p.exists()), None)
    if input_path is None:
        raise FileNotFoundError("Could not find kc_master_dataset_cleaned.parquet in candidate paths.")

    # Step 1: Phase 1 — reproject coordinates in-memory
    df_projected = inspect_topology_data(input_path)

    # Step 2: Phase 2 — KNN graph
    edge_index, edge_attr = build_knn_graph(
        df_projected,
        x_col="x_proj",
        y_col="y_proj",
    )

    # Step 3: Derive feature set — same exclusion logic as xgb_baseline.py
    feature_cols = [c for c in df_projected.columns if c not in IGNORE_COLS]
    print(f"Feature count : {len(feature_cols)}")
    print(f"Features      : {feature_cols}")

    # Step 4: Build PyG Data object
    pyg_data = create_pyg_data(df_projected, edge_index, edge_attr, feature_cols)

    print("\n" + "=" * 50)
    print("  PHASE 3 SPATIAL EMBEDDINGS & PyG DATA COMPLETED")
    print("=" * 50)
    print(f"  Node feature matrix (X) : {pyg_data.x.shape}  (raw + spatial lags)")
    print(f"  Coordinates (pos)       : {pyg_data.pos.shape}")
    print(f"  Target (Y)              : {pyg_data.y.shape}")
    print(f"  Edge index              : {pyg_data.edge_index.shape}")
    print(f"  Edge attributes         : {pyg_data.edge_attr.shape}")
    print("=" * 50)

    out_path = script_dir.parent.parent / "dataset" / "processed" / "pyg_spatial_graph.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pyg_data, out_path)
    print(f"✓ Saved PyTorch Geometric Data object to: {out_path}")
