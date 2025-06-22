# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 08:42:12 2025

@author: gabri
"""
from torch_geometric.nn import GCNConv
from torch.nn import Linear, functional as F, Module


class GCNConv_simple(Module):
    def __init__(
        self, name, input_node_dim, input_edge_dim, hidden_dim, output_node_dim
    ):
        super().__init__()
        self.name = name
        self.conv1 = GCNConv(input_node_dim, 2 * hidden_dim)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim)
        self.linear = Linear(hidden_dim, output_node_dim)

    def forward(self, x, edge_attrs, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)
        x = self.linear(x)
        return x
