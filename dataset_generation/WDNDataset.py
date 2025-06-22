# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:02:07 2025

@author: gabri
"""
import os
import wntr
import zarr
import torch

import logging
import numpy as np

from tqdm import tqdm
from time import time
import torch_geometric
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Log critical parts of code
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# logging.basicConfig(
#     filename=f"WDNDataset_{timestamp}.log",
#     encoding="utf-8",
#     level=logging.DEBUG,
# )

# logger = logging.getLogger(__name__)


class WDNDataset:
    def __init__(
        self,
        name,
        logger,
        scenarios_path,
        num_scenarios,
        mean,
        std,
        min,
        max,
        do_norm,  # True, False
        norm_type,  # znorm or minmax
        batch_size,
        keep_elements=["all"],  # [individual nodes]
        input_data="node",  # ["node", "edge", "both"]
        output_data="node",  # ["node", "edge", "both"] Always Node
        input_node_features=[
            "pressure"
        ],  # Check scene generation json file before assigning
        output_node_features=["pressure"],
        input_edge_features=["flowrate"],
        output_edge_features=["flowrate"],
    ):
        self.name = name

        self.scenarios_path = os.path.join(scenarios_path, "scenes", self.name)
        # print(self.scenarios_path)
        self.batch_size = batch_size

        # Logger Setup
        self.logger = logger
        self.logger.info(f"WDNDataset instance with name: {self.name}")

        self.input_node_features = (
            input_node_features if input_data in ["node", "both"] else None
        )
        self.output_node_features = (
            output_node_features if output_data in ["node", "both"] else None
        )
        self.input_edge_features = (
            input_edge_features if input_data in ["edge", "both"] else None
        )
        self.output_edge_features = (
            output_edge_features if output_data in ["edge", "both"] else None
        )
        static_edge_index = zarr.open(
            self.scenarios_path + "/static_edge_index.zarr", mode="r"
        )

        raw_input_node_data = None
        raw_input_edge_data = None
        raw_output_data = None

        self.logger.info(f"Input Data: {input_data}")

        if input_data == "node" or input_data == "both":
            raw_input_node_data = zarr.open(
                self.scenarios_path + "/input_node_features.zarr", mode="r"
            )
            self.logger.debug(
                f"Input Data shape: {raw_input_node_data.shape}\n"
            )

            input_node_indices = raw_input_node_data.attrs["node_indices"]
            self.logger.debug(f"Mapped node indices: {input_node_indices}\n")
            input_node_feature_indices = raw_input_node_data.attrs[
                "feature_indices"
            ]

            if "all" in keep_elements:
                keep_elements = [
                    k for k in raw_input_node_data.attrs["node_indices"].keys()
                ]

                self.logger.debug(f"Elements to keep: {keep_elements}")

            input_node_idx_keep = [
                node_idx
                for node_id, node_idx in input_node_indices.items()
                if node_id in keep_elements
            ]

            input_node_feature_idx_keep = [
                feature_idx
                for feature_name, feature_idx in input_node_feature_indices.items()
                if feature_name in input_node_features
            ]

            self.logger.debug(
                f"(INPUT) Keep node indices: {input_node_idx_keep}\n"
            )
            self.logger.debug(
                f"(INPUT) Keep node feature indices: {input_node_feature_idx_keep}\n"
            )
        elif input_data == "edge" or input_data == "both":
            raw_input_edge_data = zarr.open(
                self.scenarios_path + "/input_edge_features.zarr", mode="r"
            )
            self.logger.debug(
                f"Input Data shape: {raw_input_edge_data.shape}\n"
            )
            input_edge_indices = raw_input_edge_data.attrs["edge_indices"]

            input_edge_feature_indices = raw_input_edge_data.attrs[
                "feature_indices"
            ]
            edge_idx_keep = [
                edge_idx
                for edge_id, edge_idx in input_edge_indices.items()
                if edge_id in keep_elements
            ]

            edge_feature_idx_keep = [
                feature_idx
                for feature_name, feature_idx in input_edge_feature_indices.items()
                if feature_name in input_edge_features
            ]

            self.logger.debug(f"(INPUT) Keep edge indices: {edge_idx_keep}\n")
            self.logger.debug(
                f"(INPUT) Keep edge feature indices: {edge_feature_idx_keep}\n"
            )

        self.logger.info(f"Output Data: {output_data}")
        if output_data == "node":
            raw_output_data = zarr.open(
                self.scenarios_path + "/output_node_features.zarr", mode="r"
            )
            self.logger.debug(
                f"Node Output Data shape: {raw_output_data.shape}\n"
            )
            output_element_indices = raw_output_data.attrs["node_indices"]
            output_elements_idx_keep = [
                node_idx
                for node_id, node_idx in output_element_indices.items()
                if node_id in keep_elements
            ]

        if output_data == "edge":
            raw_output_data = zarr.open(
                self.scenarios_path + "/output_edge_features.zarr", mode="r"
            )
            self.logger.debug(
                f"Edge Output Data shape: {raw_output_data.shape}\n"
            )
            output_element_indices = raw_output_data.attrs["edge_indices"]
            if "all" in keep_elements:
                keep_elements = [
                    k for k in raw_output_data.attrs["edge_indices"].keys()
                ]

                self.logger.debug(f"Elements to keep: {keep_elements}")

        output_element_feature_indices = raw_output_data.attrs[
            "feature_indices"
        ]

        output_elements_feature_idx_keep = [
            feature_idx
            for feature_name, feature_idx in output_element_feature_indices.items()
            if feature_name in output_node_features
        ]

        self.logger.debug(
            f"(OUTPUT) Keep {output_data} indices: {output_elements_idx_keep}\n"
        )
        self.logger.debug(
            f"(OUTPUT) Keep {output_data} feature indices: {output_elements_feature_idx_keep}\n"
        )

        # Gather the flattened array
        if not (
            isinstance(keep_elements, list)
            and all(isinstance(v, str) for v in keep_elements)
        ):  # is not list[str] or str
            self.logger.debug(
                f"keep_elements type: {type(keep_elements)}, {len(keep_elements)}, {keep_elements[0]}\n"
            )
            raise ValueError("keep_elements must be List[str] or str!")

        if None in [mean, std, min, max]:
            flatten_array = []
            if raw_input_node_data is not None:
                flatten_array.append(
                    np.ravel(
                        raw_input_node_data.oindex[
                            :,
                            input_node_idx_keep,
                            input_node_feature_idx_keep,  # oindex expects list
                        ]
                    )
                )

            if raw_input_edge_data is not None:
                flatten_array.append(
                    np.ravel(
                        raw_input_edge_data.oindex[
                            :, edge_idx_keep, edge_feature_idx_keep
                        ]
                    )
                )

            flatten_array.append(
                np.ravel(
                    raw_output_data.oindex[
                        :,
                        output_elements_idx_keep,
                        output_elements_feature_idx_keep,
                    ]
                )
            )
            flatten_array = np.concatenate(flatten_array)
            self.logger.debug(f"Flatten array len: {len(flatten_array)}\n")

            self._mean = np.mean(flatten_array)
            self.logger.debug(f"Flatten array MEAN: {self.mean}\n")
            self._std = np.std(flatten_array)
            self.logger.debug(f"Flatten array STD: {self.std}\n")
            self._min = np.min(flatten_array)
            self.logger.debug(f"Flatten array MIN: {self.min}\n")
            self._max = np.max(flatten_array)
            self.logger.debug(f"Flatten array MAX: {self.max}\n")
        else:
            self._mean = mean
            self._std = std
            self._min = min
            self._max = max

        if do_norm:
            if norm_type == "znorm":
                norm_fn = self.znorm
            elif norm_type == "minmax":
                norm_fn = self.minmax
        else:
            norm_fn == self.nonorm

        self._dataset = []

        if num_scenarios > raw_input_node_data.shape[0]:
            num_scenarios = raw_input_node_data.shape[0]

        for sample_idx in range(num_scenarios):
            x = torch.tensor(
                # Normalize the data
                norm_fn(
                    raw_input_node_data.oindex[
                        sample_idx,
                        input_node_idx_keep,
                        input_node_feature_idx_keep,
                    ]
                ),
                dtype=torch.float32,
            )

            edge_attr = (
                torch.tensor(
                    norm_fn(
                        raw_input_edge_data.oindex[
                            sample_idx, edge_idx_keep, edge_feature_idx_keep
                        ]
                    ),
                    dtype=torch.float32,
                )
                if raw_input_edge_data is not None
                else None
            )

            y = torch.tensor(
                norm_fn(
                    raw_output_data.oindex[
                        sample_idx,
                        output_elements_idx_keep,
                        output_elements_feature_idx_keep,
                    ]
                ),
                dtype=torch.float32,
            )

            self._dataset.append(
                Data(
                    x=x,
                    y=y,
                    edge_attr=edge_attr,
                    edge_index=torch.tensor(static_edge_index[:]),
                )
            )
        self.logger.debug(
            f"static_edge_index.shape: {static_edge_index.shape}\n"
        )
        # self.logger.debug(f"static_edge_index: {static_edge_index}")
        self.logger.debug(f"x.shape: {x.shape}\n")
        # self.logger.debug(f"x: {x}")
        self.logger.debug(f"y.shape: {y.shape}\n")
        # self.logger.debug(f"y: {y}")
        if edge_attr is not None:
            self.logger.debug(f"edge_attr.shape: {edge_attr.shape}\n")
        else:
            self.logger.debug("edge_attr.shape: None\n")
        # self.logger.debug(f"edge_attr: {edge_attr}\n")
        self.logger.debug(
            f"Last data object: {Data(x=x,y=y, edge_attr=edge_attr, edge_index=static_edge_index)}"
        )

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def znorm(self, x):
        return (x - self.mean) / self.std

    def minmax(self, x, min, max):
        return (x - self.min) / (self.max - self.min)

    def nonorm(self, x):
        return x

    def generate_DataLoader(self):
        return DataLoader(self._dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dataset = WDNDataset(
        name="DummyTest",
        scenarios_path="network_scenarios/Train_dummy_DELETE_THIS",
        num_scenarios=10,
        mean=None,
        std=None,
        min=None,
        max=None,
        do_norm=True,
        norm_type="znorm",
        batch_size=1,
        # keep_elements="all",
    )
    print("Done")
