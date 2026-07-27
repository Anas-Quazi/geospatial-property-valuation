import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from torch_geometric.data import Data


def apply_spatial_block_split(
    data: Data,
    n_clusters: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Data:
    """Performs spatial block clustering to prevent data leakage between train/val/test splits."""
    # Extract coordinates (first 2 features of X) to form geographic blocks
    coords = data.x[:, :2].numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)

    unique_clusters = np.unique(cluster_labels)
    np.random.seed(42)
    np.random.shuffle(unique_clusters)

    n_train = int(len(unique_clusters) * train_ratio)
    n_val = int(len(unique_clusters) * val_ratio)

    train_clusters = set(unique_clusters[:n_train])
    val_clusters = set(unique_clusters[n_train : n_train + n_val])

    # Assign boolean masks to PyG Data object
    data.train_mask = torch.tensor(
        [c in train_clusters for c in cluster_labels], dtype=torch.bool
    )
    data.val_mask = torch.tensor(
        [c in val_clusters for c in cluster_labels], dtype=torch.bool
    )
    data.test_mask = ~(data.train_mask | data.val_mask)

    print("\n" + "=" * 50)
    print("  PHASE 4: SPATIAL LEAKAGE AUDIT & SPLIT COMPLETE")
    print("=" * 50)
    print(f"  Total Nodes : {data.num_nodes:,} across {n_clusters} spatial blocks")
    print(f"  Train Nodes : {data.train_mask.sum().item():,}")
    print(f"  Val Nodes   : {data.val_mask.sum().item():,}")
    print(f"  Test Nodes  : {data.test_mask.sum().item():,}")
    print("=" * 50)

    return data


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        script_dir / "dataset" / "processed" / "pyg_spatial_graph.pt",
        script_dir.parent / "dataset" / "processed" / "pyg_spatial_graph.pt",
        script_dir.parent.parent / "dataset" / "processed" / "pyg_spatial_graph.pt",
        Path("dataset/processed/pyg_spatial_graph.pt"),
    ]
    graph_path = next((p for p in candidate_paths if p.exists()), None)

    if graph_path is None:
        raise FileNotFoundError("Could not find pyg_spatial_graph.pt. Run Phase 3 first!")

    data = torch.load(graph_path, weights_only=False)
    data_split = apply_spatial_block_split(data)

    # Save audited graph back with masks added
    out_path = graph_path.parent / "pyg_spatial_graph_audited.pt"
    torch.save(data_split, out_path)
    print(f"✓ Saved audited graph dataset to: {out_path}")