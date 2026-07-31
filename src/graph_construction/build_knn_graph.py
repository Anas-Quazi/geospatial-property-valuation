import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Phase 2: KNN Graph Construction
# ---------------------------------------------------------------------------
# Builds a directed K-nearest-neighbor spatial graph from projected coords.
# Owner : Aadi-1605
# Input : dataset/processed/kc_master_dataset_cleaned.parquet
#         (requires x_coords/y_coords or projected_x/projected_y from Phase 1)
# ---------------------------------------------------------------------------

K_NEIGHBORS = 10
SIGMA = 500.0 


def build_knn_graph(df: pd.DataFrame, k: int = K_NEIGHBORS, sigma: float = SIGMA,
                     x_col="x_coords", y_col="y_coords"):
    """
    Builds a directed KNN graph over projected coordinates.

    Returns edge_index [2, E] (source, dest node ids) and edge_attr [E, 1]
    (Gaussian-decay weights, bounded in (0, 1]). Node ids correspond to
    `df['id']`, which is assumed to align with df's positional row order (0..n-1).
    """
    df_copy = df.copy()

    assert df_copy["id"].is_unique, "id column must be unique per node"
    assert (df_copy["id"].values == np.arange(len(df_copy))).all(), \
        "id must match positional row index for edge_index to reference nodes correctly"
    assert df_copy[[x_col, y_col]].isna().sum().sum() == 0, "null coordinates found"

    coords = df_copy[[x_col, y_col]].to_numpy()
    tree = KDTree(coords)

    
    dists, idxs = tree.query(coords, k=k + 1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  

    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = idxs.reshape(-1)
    edge_index = np.stack([src, dst], axis=0)  

    d = dists.reshape(-1)
    edge_attr = np.exp(-(d ** 2) / (2 * sigma ** 2)).reshape(-1, 1)  

    return edge_index, edge_attr


def to_undirected_np(edge_index: np.ndarray, edge_attr: np.ndarray):
    """
    Optional: symmetrizes a directed edge_index/edge_attr pair by adding
    reverse edges and dropping duplicates, matching the behavior of
    torch_geometric.utils.to_undirected. Pure numpy — no torch dependency.

    If torch_geometric is available, the equivalent call is:
        from torch_geometric.utils import to_undirected
        edge_index_u, edge_attr_u = to_undirected(
            torch.from_numpy(edge_index), torch.from_numpy(edge_attr), reduce="mean"
        )
    """
    src, dst = edge_index
    rev_src, rev_dst = dst, src

    all_src = np.concatenate([src, rev_src])
    all_dst = np.concatenate([dst, rev_dst])
    all_attr = np.concatenate([edge_attr, edge_attr], axis=0)

    
    pairs = np.stack([all_src, all_dst], axis=1)
    uniq_pairs, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)

    summed = np.zeros((uniq_pairs.shape[0], all_attr.shape[1]))
    np.add.at(summed, inverse, all_attr)
    averaged = summed / counts[:, None]

    edge_index_u = uniq_pairs.T
    edge_attr_u = averaged
    return edge_index_u, edge_attr_u


def save_graph(path: Path, edge_index: np.ndarray, edge_attr: np.ndarray):
    np.savez(path, edge_index=edge_index, edge_attr=edge_attr)

# Phase 2 Main Block (Importing Phase 1 dynamically)
if __name__ == "__main__":
    from build_spatial_topology import inspect_topology_data

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

    # 1. Run Phase 1 in-memory to project coordinates & ensure node IDs
    df_projected = inspect_topology_data(input_path)

    # 2. Feed the returned DataFrame directly into Phase 2 graph construction
    edge_index, edge_attr = build_knn_graph(
        df_projected, 
        k=K_NEIGHBORS, 
        x_col="x_proj", 
        y_col="y_proj"
    )

    print(f"Nodes: {df_projected.shape[0]:,}")
    print(f"Edges (directed): {edge_index.shape[1]:,}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_attr shape: {edge_attr.shape}")

    # 3. Save the final graph output artifact (.npz)
    out_path = script_dir.parent.parent / "dataset" / "processed" / "knn_graph.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(out_path, edge_index, edge_attr)
    print(f"Saved KNN graph to: {out_path}")
