# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:25:01 2025

@author: gabri
"""
import zarr
import wntr
import json
import torch
import logging
import numpy as np

from tqdm import tqdm
from time import time
from datetime import datetime
from collections import defaultdict

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"SceneGenerator_{timestamp}.log",
    encoding="utf-8",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class SceneGenerator:
    def __init__(
        self,
        base_path,
        name,
        inp_file,
        config_file,
        timestep,
        num_scenarios,
        pipe_open_prob,
        pump_open_prob,
        valve_open_prob,
        remove_controls,
        input_node_features,
        output_node_features,
        input_edge_features,
        output_edge_features,
        random_seed,
    ):
        # Set the random seed
        np.random.seed(random_seed)

        self.name = name
        self.invalid_simulations = 0  # For debugging purposes
        # Logger Setup
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}.{self.name}\n"
        )

        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"Class created at: {timestamp}\n")

        self.num_scenarios = num_scenarios

        # Initialized as dict to enable retrieval of specific features later by the WDNDataset class
        self._input_node_features = {
            feature_name: fi
            for fi, feature_name in enumerate(input_node_features)
        }
        self.logger.info(f"(INPUT) Node features: {self._input_node_features}")

        self._input_edge_features = {
            feature_name: fi
            for fi, feature_name in enumerate(input_edge_features)
        }
        self.logger.info(f"(INPUT) Edge features: {self._input_edge_features}")

        self._output_node_features = {
            feature_name: fi
            for fi, feature_name in enumerate(output_node_features)
        }
        self.logger.info(
            f"(OUTPUT) Node features: {self._output_node_features}"
        )

        self._output_edge_features = {
            feature_name: fi
            for fi, feature_name in enumerate(output_edge_features)
        }
        self.logger.info(
            f"(OUTPUT) Edge features: {self._output_edge_features}"
        )

        # Pipe, Pump and Valve open probabilities
        self._pipe_open_prob = pipe_open_prob
        self._pump_open_prob = pump_open_prob
        self._valve_open_prob = valve_open_prob

        # Open the json configuration file
        with open(config_file, "r") as jfile:
            self.config_file = json.load(jfile)

        # Define the Water Network Model
        self.logger.info("Setting up Water Network Model")
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.wn.options.time.duration = timestep
        self.wn.options.time.hydraulic_timestep = timestep
        self.wn.options.time.pattern_timestep = timestep
        self.wn.options.hydraulic.demand_model = "DD"

        # Disable controls
        if remove_controls:
            for ctrl_name in self.wn.control_name_list:
                self.wn.remove_control(ctrl_name)

        # Edge index
        self.node_id_map = {
            node_id: i for i, node_id in enumerate(self.wn.node_name_list)
        }
        self.link_id_map = {
            link_id: i for i, link_id in enumerate(self.wn.link_name_list)
        }
        self.logger.info(f"Node mappings: {self.node_id_map}")
        self.logger.info(f"Link mappings: {self.link_id_map}")

        edge_list = []
        for link_id in self.link_id_map.keys():
            link = self.wn.get_link(link_id)
            start_node_id, end_node_id = (
                link.start_node_name,
                link.end_node_name,
            )
            ui, vi = (
                self.node_id_map[start_node_id],
                self.node_id_map[end_node_id],
            )
            edge_list.append([ui, vi])
            edge_list.append([vi, ui])

        self.static_edge_index = (
            torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        )

        # Scenario Storage
        self.logger.info(f"Generating zarrays at {base_path}/scenes")
        zarr.save(
            f"{base_path}/scenes/{self.name}/static_edge_index.zarr",
            self.static_edge_index.detach().numpy(),
        )

        self.input_node_feature_storage = zarr.create(
            store=f"{base_path}/scenes/{self.name}/input_node_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.node_id_map.keys()),
                len(self._input_node_features.keys()),
            ),
            # chunks=(100, len(self.input_node_features)),
            dtype="f4",
            compressor=zarr.codecs.Blosc(
                cname="zstd",
                clevel=1,
                shuffle=zarr.codecs.Blosc.SHUFFLE,
            ),
            overwrite=True,
        )
        self.input_node_feature_storage.attrs[
            "feature_indices"
        ] = self._input_node_features
        self.input_node_feature_storage.attrs[
            "node_indices"
        ] = self.node_id_map

        self.input_edge_feature_storage = zarr.create(
            store=f"{base_path}/scenes/{self.name}/input_edge_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.link_id_map.keys()),
                len(self._input_edge_features.keys()),
            ),
            # chunks=(100, len(self.input_node_features)),
            dtype="f4",
            compressor=zarr.codecs.Blosc(
                cname="zstd",
                clevel=1,
                shuffle=zarr.codecs.Blosc.SHUFFLE,
            ),
            overwrite=True,
        )
        self.input_edge_feature_storage.attrs[
            "feature_indices"
        ] = self._input_edge_features
        self.input_node_feature_storage.attrs[
            "edge_indices"
        ] = self.link_id_map

        self.output_node_feature_storage = zarr.create(
            store=f"{base_path}/scenes/{self.name}/output_node_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.node_id_map.keys()),
                len(self._output_node_features.keys()),
            ),
            # chunks=(100, len(self.output_node_features)),
            dtype="f4",
            compressor=zarr.codecs.Blosc(
                cname="zstd",
                clevel=1,
                shuffle=zarr.codecs.Blosc.SHUFFLE,
            ),
            overwrite=True,
        )
        self.output_node_feature_storage.attrs[
            "feature_indices"
        ] = self._output_node_features
        self.output_node_feature_storage.attrs[
            "node_indices"
        ] = self.node_id_map

        self.output_edge_feature_storage = zarr.create(
            store=f"{base_path}/scenes/{self.name}/output_edge_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.link_id_map.keys()),
                len(self._output_edge_features.keys()),
            ),
            # chunks=(100, len(self.output_node_features)),
            dtype="f4",
            compressor=zarr.codecs.Blosc(
                cname="zstd",
                clevel=1,
                shuffle=zarr.codecs.Blosc.SHUFFLE,
            ),
            overwrite=True,
        )
        self.output_edge_feature_storage.attrs[
            "feature_indices"
        ] = self._output_edge_features
        self.output_edge_feature_storage.attrs[
            "edge_indices"
        ] = self.link_id_map

    def generate_input_snapshot(self, scene_i):
        """
        Generates a random scenarion of the WDN and saves the base demand and base head for
        each junction/reservoir to be used to generate the output scenario
        """
        output_snpsht_values = defaultdict(
            dict
        )  # The dictionary containing the difference values needed to generate the output snapshot

        # Junctions
        for junction_id, junction_data in self.config_file[
            "junctions"
        ].items():
            junction = self.wn.get_node(junction_id)

            random_demand = np.random.uniform(
                low=self.config_file["junctions"][junction_id][
                    "junc_demand_lo"
                ],
                high=self.config_file["junctions"][junction_id][
                    "junc_demand_hi"
                ],
            )

            junction.demand_timeseries_list[0].base_value = random_demand

            output_snpsht_values[junction_id][
                "input_base_demand"
            ] = random_demand
            junction.demand_timeseries_list[0].pattern_name = None

        # Pipes
        for pipe_id, pipe_data in self.config_file["pipes"].items():
            status = np.random.choice(
                [1, 0],
                p=[self._pipe_open_prob, 1.0 - self._pipe_open_prob],
            )
            pipe = self.wn.get_link(pipe_id)
            pipe.initial_status = status

        # Pumps
        for pump_id, pump_data in self.config_file["pumps"].items():
            status = np.random.choice(
                [1, 0], p=[self._pump_open_prob, 1.0 - self._pump_open_prob]
            )
            pump = self.wn.get_link(pump_id)
            pump.initial_status = status

        # Valves
        for valve_id, valve_data in self.config_file["valves"].items():
            status = np.random.choice(
                [1, 0], p=[self._valve_open_prob, 1.0 - self._valve_open_prob]
            )
            valve = self.wn.get_link(valve_id)
            valve.initial_status = status

        # Reservoirs
        for reservoir_id, reservoir_data in self.config_file[
            "reservoirs"
        ].items():
            reservoir = self.wn.get_node(reservoir_id)
            if (
                self.config_file["reservoirs"][reservoir_id]["res_head_hi"]
                is not None
                and self.config_file["reservoirs"][reservoir_id]["res_head_lo"]
                is not None
            ):
                random_head = np.random.uniform(
                    low=self.config_file["reservoirs"][reservoir_id][
                        "res_head_lo"
                    ],
                    high=self.config_file["reservoirs"][reservoir_id][
                        "res_head_hi"
                    ],
                )

                reservoir.head_timeseries.base_value = random_head

                output_snpsht_values[reservoir_id][
                    "input_base_head"
                ] = random_head
                reservoir.head_timeseries.pattern_name = None

        # Run simulation
        sim = wntr.sim.EpanetSimulator(self.wn)
        results_input = sim.run_sim()

        return results_input, output_snpsht_values

    def generate_output_snapshot(self, input_values):
        """
        Generate WDN scenario using the input base demand and input base head and adding a
        randomly sampled difference of demand and head
        """
        # Junctions
        for junction_id, junction_data in self.config_file[
            "junctions"
        ].items():
            junction = self.wn.get_node(junction_id)

            random_diff_demand = np.random.uniform(
                low=self.config_file["junctions"][junction_id][
                    "junc_demand_diff_lo"
                ],
                high=self.config_file["junctions"][junction_id][
                    "junc_demand_diff_hi"
                ],
            )

            junction.demand_timeseries_list[0].base_value = (
                input_values[junction_id]["input_base_demand"]
                + random_diff_demand
            )
            junction.demand_timeseries_list[0].pattern_name = None

        # Pipes
        for pipe_id, pipe_data in self.config_file["pipes"].items():
            status = np.random.choice(
                [1, 0],
                p=[self._pipe_open_prob, 1.0 - self._pipe_open_prob],
            )
            pipe = self.wn.get_link(pipe_id)
            pipe.initial_status = status

        # Pumps
        for pump_id, pump_data in self.config_file["pumps"].items():
            status = np.random.choice(
                [1, 0], p=[self._pump_open_prob, 1.0 - self._pump_open_prob]
            )
            pump = self.wn.get_link(pump_id)
            pump.initial_status = status

        # Valves
        for valve_id, valve_data in self.config_file["valves"].items():
            status = np.random.choice(
                [1, 0],
                p=[self._valve_open_prob, 1.0 - self._valve_open_prob],
            )
            valve = self.wn.get_link(valve_id)
            valve.initial_status = status

        # Reservoirs
        for reservoir_id, reservoir_data in self.config_file[
            "reservoirs"
        ].items():
            reservoir = self.wn.get_node(reservoir_id)
            if (
                self.config_file["reservoirs"][reservoir_id]["res_head_hi"]
                is not None
                and self.config_file["reservoirs"][reservoir_id]["res_head_lo"]
                is not None
            ):
                random_diff_head = np.random.uniform(
                    low=self.config_file["reservoirs"][reservoir_id][
                        "res_diff_head_lo"
                    ],
                    high=self.config_file["reservoirs"][reservoir_id][
                        "res_diff_head_hi"
                    ],
                )
                reservoir.head_timeseries.base_value = (
                    input_values[reservoir_id]["input_base_head"]
                    + random_diff_head
                )
                reservoir.head_timeseries.pattern_name = None

        # Run simulation
        sim = wntr.sim.EpanetSimulator(self.wn)
        results_output = sim.run_sim()

        # Reset the configured network
        self.wn.reset_initial_values()
        return results_output

    def _generate_one_sample(self, tmp_scene_id):
        """
        Generate one input -> output sample
        """
        # Generate the napshots
        results_input, input_values = self.generate_input_snapshot(
            tmp_scene_id
        )
        results_output = self.generate_output_snapshot(input_values)

        # Gather Node features for single snapshot
        input_node_features_values = []
        output_node_features_values = []
        for node_name in self.node_id_map.keys():
            tmp_input_node_features = []

            for input_node_feature in self._input_node_features.keys():
                tmp_input_node_features.append(
                    results_input.node[input_node_feature][node_name].values[0]
                )
            input_node_features_values.append(tmp_input_node_features)

            tmp_output_node_features = []
            for output_node_feature in self._output_node_features.keys():
                tmp_output_node_features.append(
                    results_output.node[output_node_feature][node_name].values[
                        0
                    ]
                )
            output_node_features_values.append(tmp_output_node_features)

        # Gather Link features for single snapshot
        input_edge_features_values = []
        output_edge_features_values = []
        for link_name, link_idx in self.link_id_map.items():
            tmp_input_edge_features = []
            for input_edge_feature in self._input_edge_features.keys():
                tmp_input_edge_features.append(
                    results_input.link[input_edge_feature][link_name].values[0]
                )
            input_edge_features_values.append(tmp_input_edge_features)

            tmp_output_edge_features = []
            for output_edge_feature in self._output_edge_features.keys():
                tmp_output_edge_features.append(
                    results_output.link[output_edge_feature][link_name].values[
                        0
                    ]
                )
            output_edge_features_values.append(tmp_output_edge_features)

        (
            input_node_features,
            output_node_features,
            input_edge_features,
            output_edge_features,
        ) = (
            np.vstack(input_node_features_values),
            np.vstack(output_node_features_values),
            np.vstack(input_edge_features_values),
            np.vstack(output_edge_features_values),
        )

        return (
            input_node_features,
            output_node_features,
            input_edge_features,
            output_edge_features,
        )

    def generate_scenarios(self):
        start_time = time()
        # minimal_value = -100

        scene_i = 0
        pbar = tqdm(total=self.num_scenarios, desc="Generating scenarios")
        while scene_i < self.num_scenarios:
            (
                input_node_features,
                output_node_features,
                input_edge_features,
                output_edge_features,
            ) = self._generate_one_sample(scene_i)

            if np.any(
                input_node_features[self._input_node_features["pressure"]] < 0
            ):
                # print("Resampling scene: ", scene_i)
                # print("Negative pressure values in input snapshot")
                self.invalid_simulations += 1
                continue

            if np.any(
                output_node_features[self._input_node_features["pressure"]] < 0
            ):
                # print("Resampling scene: ", scene_i)
                # print("Negative pressure values in output snapshot")
                self.invalid_simulations += 1
                continue

            # Store the different aspects of the samples
            self.input_node_feature_storage[
                scene_i, :, :
            ] = input_node_features
            self.output_node_feature_storage[
                scene_i, :, :
            ] = output_node_features
            self.input_edge_feature_storage[
                scene_i, :, :
            ] = input_edge_features
            self.output_edge_feature_storage[
                scene_i, :, :
            ] = output_edge_features

            scene_i += 1
            pbar.update(1)

        pbar.close()
        print(f"Generation completed in {time() - start_time:.2f} seconds")
        self.logger.debug(
            f"Number of invalid simulations: {self.invalid_simulations}"
        )
