import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class SpatialGraphSAGE(nn.Module):
    """2-Layer GraphSAGE model for spatial property price regression."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr="mean")
        self.fc = nn.Linear(hidden_channels // 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return self.fc(x)


def train_and_evaluate(data: Data, epochs: int = 150, lr: float = 0.005):
    """Trains GraphSAGE and measures property price predictions in real dollars."""
    y_target = torch.log1p(data.y)  # Log-transform for scale stability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatialGraphSAGE(in_channels=data.x.size(1)).to(device)
    data = data.to(device)
    y_target = y_target.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"\n🚀 Training GraphSAGE Model on Device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], y_target[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(
                    val_out[data.val_mask], y_target[data.val_mask]
                ).item()
            print(
                f"Epoch {epoch:03d} | Train Loss (MSE): {loss.item():.4f} | Val Loss (MSE): {val_loss:.4f}"
            )

    # Test set evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)

        # Convert predictions & ground truth back to dollar values
        preds = torch.expm1(test_out[data.test_mask]).cpu().numpy()
        actuals = torch.expm1(y_target[data.test_mask]).cpu().numpy()

        mae = np.mean(np.abs(preds - actuals))
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))

    print("\n" + "=" * 50)
    print("  PHASE 5: FINAL GNN MODEL EVALUATION (TEST SET)")
    print("=" * 50)
    print(f"  Mean Absolute Error (MAE)  : ${mae:,.2f}")
    print(f"  Root Mean Sq. Error (RMSE) : ${rmse:,.2f}")
    print(f"  Mean Abs % Error (MAPE)   : {mape:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        script_dir / "dataset" / "processed" / "pyg_spatial_graph_audited.pt",
        script_dir / "dataset" / "processed" / "pyg_spatial_graph.pt",
        script_dir.parent / "dataset" / "processed" / "pyg_spatial_graph_audited.pt",
        script_dir.parent.parent / "dataset" / "processed" / "pyg_spatial_graph_audited.pt",
    ]
    graph_path = next((p for p in candidate_paths if p.exists()), None)

    if graph_path is None:
        raise FileNotFoundError("Could not find dataset. Run Phase 3 & 4 first!")

    data = torch.load(graph_path)

    # Apply spatial split if masks aren't already generated
    if not hasattr(data, "train_mask"):
        from phase4_spatial_audit import apply_spatial_block_split
        data = apply_spatial_block_split(data)

    train_and_evaluate(data)