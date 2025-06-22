# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:49:52 2025

@author: gabri
"""
from .graph_models import GCNConv_simple


def gcnconv_simple_16h(args):
    return args, GCNConv_simple(
        name="GCNConv_simple_16h",
        input_node_dim=len(args.input_node_features),
        input_edge_dim=None,
        hidden_dim=16,
        output_node_dim=len(args.output_node_features),
    )


def select_model(args):
    assert args.model_config in [
        "gcnconv_simple_16h"
    ], "Model configuration not present. Add model configuration in 'model_configurations.py'"
    if args.model_config == "gcnconv_simple_16h":
        args, model = gcnconv_simple_16h(args)
        return args, model
