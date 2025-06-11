# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:12:34 2025

@author: gabri
"""
from torch import nn
from torch.nn import GRU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal import TGCN


class TGCNForecast(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # TGCN layer: integrates GCN and GRU logic
        self.tgcn1 = TGCN(in_channels, hidden_channels)
        self.tgcn2 = TGCN(hidden_channels, hidden_channels)
        # Final linear layer to map hidden states to forecasted values
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x: [num_nodes, in_channels]
        h = self.tgcn1(x, edge_index, edge_weight)
        h = self.tgcn2(h, edge_index, edge_weight)
        return self.linear(h)  # outputs [num_nodes, out_channels]


class ConvGRUForecaster(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Spatial layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Temporal layers
        self.gru1 = GRU(
            hidden_channels,
            hidden_channels,
            bidirectional=False,
            batch_first=True,
        )  # batch, numnodes, infeats
        self.gru2 = GRU(
            hidden_channels,
            hidden_channels,
            bidirectional=False,
            batch_first=True,
        )  # batch, numnodes, infeats
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout2(h)
        out, gru_h = self.gru1(h)
        out, _ = self.gru2(out)
        out = self.linear(out)
        return out
