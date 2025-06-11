# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:36:12 2025

@author: gabri
"""
import sys
from tqdm import tqdm

print(sys.executable)
import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN

# Load the static graph temporal signal dataset
loader = METRLADatasetLoader()  # creates a data loader for the dataset[4]
dataset = loader.get_dataset()  # returns an iterable over time steps

# Each element in `dataset` is a snapshot with:
#   snapshot.x          Node features at current time step (shape: [num_nodes, num_features])
#   snapshot.edge_index Edge list of the static graph (shape: [2, num_edges])
#   snapshot.edge_weight (optional) Edge weights, if available
#   snapshot.y          Node targets at next time step (shape: [num_nodes, 1])


class TGCNForecast(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # TGCN layer: integrates GCN and GRU logic
        self.tgcn = TGCN(in_channels, hidden_channels)
        # Final linear layer to map hidden states to forecasted values
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [num_nodes, in_channels]
        h = self.tgcn(
            x, edge_index, edge_weight
        )  # outputs [num_nodes, hidden_channels]
        return self.linear(h)  # outputs [num_nodes, out_channels]


# %% Training
device = torch.device("cuda")
model = TGCNForecast(
    in_channels=dataset.features[0].shape[1] * dataset.features[0].shape[2],
    hidden_channels=32,
    out_channels=dataset.features[0].shape[2],
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
criterion = nn.MSELoss()

num_epochs = 50

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    # Iterate over time steps
    for snapshot in tqdm(dataset, total=dataset.snapshot_count):
        x = torch.reshape(
            snapshot.x, (207, snapshot.x.shape[1] * snapshot.x.shape[2])
        ).to(device)
        edge_index = snapshot.edge_index.to(device)
        # edge_weight = getattr(snapshot, "edge_weight", None)
        edge_weight = getattr(snapshot, "edge_weight", None)
        y_true = snapshot.y.to(device)

        # Forward pass: predict next‐step node values
        y_pred = model(x, edge_index, edge_weight)

        # Compute loss
        loss = criterion(y_pred, y_true)
        loss.backward()  # accumulate gradients
        total_loss += loss.item()

    optimizer.step()  # update parameters once per epoch
    avg_loss = total_loss / dataset.snapshot_count
    print(f"Epoch {epoch:02d}, Loss: {avg_loss:.4f}")

# %% Evaluation
model.eval()
with torch.no_grad():
    # Initialize with the last observed snapshot
    x, edge_index, edge_weight = (
        dataset.features[-1].to(device),
        dataset.edge_index.to(device),
        getattr(dataset, "edge_weight", None),
    )
    forecasts = []
    for _ in range(12):  # forecast next 12 time steps
        y_pred = model(x, edge_index, edge_weight)
        forecasts.append(y_pred.cpu())
        # use previous prediction as next input
        x = y_pred

# `forecasts` is a list of tensors [num_nodes, 1] for each future step
