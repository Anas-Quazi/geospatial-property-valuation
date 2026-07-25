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
# Owner : Aadi-1605
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
]

TARGET_COL = "price"


def compute_spatial_lags(
    x_matrix: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Computes weighted spatial lag features for each node based on its K neighbors."""
    src, dst = edge_index
    weights = edge_attr.squeeze()

    # Sparse weighted sum of neighbor features
    spatial_lags = np.zeros_like(x_matrix)
    weighted_features = x_matrix[dst] * weights[:, None]

    np.add.at(spatial_lags, src, weighted_features)

    # Normalize by total incoming weight per node
    weight_sums = np.zeros(n_nodes)
    np.add.at(weight_sums, src, weights)
    weight_sums = np.maximum(weight_sums, 1e-8)  # Avoid div by zero

    spatial_lags = spatial_lags / weight_sums[:, None]
    return spatial_lags


def create_pyg_data(
    df: pd.DataFrame,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    feature_cols: list = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> Data:
    """Assembles node features, spatial embeddings, targets, and graph structure

    into a PyTorch Geometric Data instance.
    """
    n_nodes = len(df)

    # 1. Standard Node Features (Raw Structural Specs)
    x_raw = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    # 2. Engineer Spatial Context Lag Embeddings
    spatial_lags = compute_spatial_lags(x_raw, edge_index, edge_attr, n_nodes)

    # 3. Concatenate Base Features + Spatial Lags into combined Node Matrix X
    x_combined = np.hstack([x_raw, spatial_lags]).astype(np.float32)

    # 4. Target Matrix Y (Log-transformed or raw property price)
    y = df[target_col].to_numpy(dtype=np.float32)

    # Convert everything to PyTorch Tensors
    x_tensor = torch.from_numpy(x_combined)
    y_tensor = torch.from_numpy(y).unsqueeze(1)
    edge_index_tensor = torch.from_numpy(edge_index).to(torch.long)
    edge_attr_tensor = torch.from_numpy(edge_attr).to(torch.float32)

    # Construct PyTorch Geometric Data object
    data = Data(
        x=x_tensor,
        y=y_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
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

    # Step 1: Run Phase 1 in-memory (Coordinates reprojection)
    df_projected = inspect_topology_data(input_path)

    # Step 2: Run Phase 2 in-memory (KNN Graph construction)
    edge_index, edge_attr = build_knn_graph(
    df_projected,
    x_col="x_proj",
    y_col="y_proj",
)

    # Step 3: Run Phase 3 (Spatial Lags & PyG Data creation)
    pyg_data = create_pyg_data(df_projected, edge_index, edge_attr)

    print("\n" + "=" * 50)
    print("  PHASE 3 SPATIAL EMBEDDINGS & PyG DATA COMPLETED")
    print("=" * 50)
    print(f"  PyG Data Object        : {pyg_data}")
    print(f"  Node Feature Matrix (X): {pyg_data.x.shape} (Base + Spatial Lags)")
    print(f"  Target Matrix (Y)      : {pyg_data.y.shape}")
    print(f"  Edge Index             : {pyg_data.edge_index.shape}")
    print(f"  Edge Attributes        : {pyg_data.edge_attr.shape}")
    print("=" * 50)

    # Save the PyG graph dataset artifact
    out_path = script_dir.parent.parent / "dataset" / "processed" / "pyg_spatial_graph.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pyg_data, out_path)
    print(f"✓ Saved PyTorch Geometric Data object to: {out_path}")