import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class SpatialGraphGAT(nn.Module):
    """2-Layer GAT model for spatial property price regression."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, heads: int = 8):
        super().__init__()
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # Second GAT layer aggregating multi-head features
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=4, concat=False)

        # Dense MLP Head with Skip Connection for node features
        self.fc1 = nn.Linear(hidden_channels + in_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        # Graph Attention Layers
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = F.elu(h)

        # Skip Connection: Concatenate graph spatial context with raw home features
        h_combined = torch.cat([h, x], dim=-1)

        out = F.relu(self.fc1(h_combined))
        out = self.dropout(out)
        return self.fc2(out)


def train_and_evaluate(
    data: Data,
    epochs: int = 200,
    lr: float = 0.005,
    patience: int = 30,
):
    """Trains the GAT model and measures property price predictions in real dollars.

    Fixes vs. the original version:
      1. CRITICAL: best_model_state is no longer reloaded every epoch. Previously the
         model was reset to the best checkpoint after every single training step, which
         meant the optimizer's momentum/variance state and the actual weights were
         fighting each other and the model could never really move. It's now loaded
         exactly once, after training completes, for final evaluation.
      2. Added a ReduceLROnPlateau scheduler tied to validation loss.
      3. Added real early stopping (with a patience window) instead of always running
         the full epoch budget.
      4. Corrected the log message (this is GAT, not GraphSAGE) and the loss label
         (Huber, not MSE).
      5. Added an epsilon guard on MAPE to avoid divide-by-zero blowups.
    """
    # 1. Feature Scaling using Train Split Statistics (unchanged - this was already
    # correctly avoiding train/val/test leakage)
    x_mean = data.x[data.train_mask].mean(dim=0, keepdim=True)
    x_std = data.x[data.train_mask].std(dim=0, keepdim=True) + 1e-6
    data.x = (data.x - x_mean) / x_std

    y_target = torch.log1p(data.y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatialGraphGAT(in_channels=data.x.size(1)).to(device)
    data = data.to(device)
    y_target = y_target.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.HuberLoss(delta=0.1)

    print(f"\n Training GAT Model on Device: {device}")

    best_val_loss = float("inf")
    best_model_state = None
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], y_target[data.train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute validation loss every epoch to track the best checkpoint
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(
                val_out[data.val_mask], y_target[data.val_mask]
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 25 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | Train Loss (Huber): {loss.item():.4f} | "
                f"Val Loss (Huber): {val_loss:.4f} | LR: {current_lr:.6f}"
            )

        # Early stopping - NOTE: we no longer reload best_model_state here every
        # epoch. We just track whether we've stalled long enough to stop.
        if epochs_since_improvement >= patience:
            print(f"\n Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Reload the best checkpoint exactly ONCE, for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test set evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)

        # Convert predictions & ground truth back to dollar values
        # Clamp log outputs to prevent exponential overflow in expm1
        test_out_clamped = torch.clamp(test_out[data.test_mask], min=10.0, max=18.0)

        preds = torch.expm1(test_out_clamped).squeeze().cpu().numpy()
        actuals = data.y.squeeze()[data.test_mask].cpu().numpy()

        mae = np.mean(np.abs(preds - actuals))
        eps = 1e-6  # guard against divide-by-zero on any near-zero actuals
        mape = np.mean(np.abs((actuals - preds) / np.clip(actuals, eps, None))) * 100
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))

    print("\n" + "=" * 50)
    print("  PHASE 5: FINAL GNN MODEL EVALUATION (TEST SET)")
    print("=" * 50)
    print(f"  Mean Absolute Error (MAE)  : ${mae:,.2f}")
    print(f"  Root Mean Sq. Error (RMSE) : ${rmse:,.2f}")
    print(f"  Mean Abs % Error (MAPE)    : {mape:.2f}%")
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

    data = torch.load(graph_path, weights_only=False)

    # Apply spatial split if masks aren't already generated
    if not hasattr(data, "train_mask"):
        from spatial_audit import apply_spatial_block_split
        data = apply_spatial_block_split(data)

    train_and_evaluate(data)