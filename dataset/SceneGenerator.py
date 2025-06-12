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

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="SceneGenerator.log", encoding="utf-8", level=logging.DEBUG
)


class SceneGenerator:
    def __init__(
        self,
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
        self.name = name
        self.num_scenarios = num_scenarios

        self.input_node_features = input_node_features
        self.input_edge_features = input_edge_features
        self.output_node_features = output_node_features
        self.output_edge_features = output_edge_features

        self.pipe_open_prob = pipe_open_prob
        self.pump_open_prob = pump_open_prob
        self.valve_open_prob = valve_open_prob

        with open(config_file, "r") as jfile:
            self.config_file = json.load(jfile)

        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.wn.options.time.duration = timestep
        self.wn.options.time.hydraulic_timestep = timestep
        self.wn.options.hydraulic.demand_model = "DD"

        # Disable controls
        if remove_controls:
            for ctrl_name in self.wn.control_name_list:
                self.wn.remove_control(ctrl_name)

        # Edge index
        # TODO enable removal of some nodes/edges
        self.node_id_map = {
            node: i for i, node in enumerate(self.wn.node_name_list)
        }
        self.link_id_map = {
            link: i for i, link in enumerate(self.wn.link_name_list)
        }

        G = self.wn.to_graph()
        edge_list = []
        for u, v in G.edges(data=False):
            ui, vi = self.node_id_map[u], self.node_id_map[v]
            edge_list.append([ui, vi])
            edge_list.append([vi, ui])

        self.static_edge_index = (
            torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        )

        np.random.seed(random_seed)

        # Scenario Storage
        zarr.save(
            f"network_scenarios/{self.name}/static_edge_index.zarr",
            self.static_edge_index.detach().numpy(),
        )

        self.input_node_feature_storage = zarr.create(
            store=f"network_scenarios/{self.name}/input_node_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.node_id_map.keys()),
                len(self.input_node_features),
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
        self.input_edge_feature_storage = zarr.create(
            store=f"network_scenarios/{self.name}/input_edge_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.link_id_map.keys()),
                len(self.input_edge_features),
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
        self.output_node_feature_storage = zarr.create(
            store=f"network_scenarios/{self.name}/output_node_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.node_id_map.keys()),
                len(self.output_node_features),
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

        self.output_edge_feature_storage = zarr.create(
            store=f"network_scenarios/{self.name}/output_edge_features.zarr",
            # mode="w",
            shape=(
                self.num_scenarios,
                len(self.link_id_map.keys()),
                len(self.output_edge_features),
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

    def generate_input_snapshot(self):
        output_snpsht_values = {}

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
                    "junc_demand_lo"
                ],
            )

            junction.demand_timeseries_list[0].base_value = random_demand

            output_snpsht_values[junction_id] = {}
            output_snpsht_values[junction_id][
                "input_base_demand"
            ] = random_demand
            junction.demand_timeseries_list[0].pattern_name = None

        # Pipes
        for pipe_id, pipe_data in self.config_file["pipes"].items():
            status = np.random.choice(
                [1, 0],
                p=[self.pipe_open_prob, 1.0 - self.pipe_open_prob],
            )
            pipe = self.wn.get_link(pipe_id)
            pipe.initial_status = status

        # Pumps
        for pump_id, pump_data in self.config_file["pumps"].items():
            status = np.random.choice(
                [1, 0], p=[self.pump_open_prob, 1.0 - self.pump_open_prob]
            )
            pump = self.wn.get_link(pump_id)
            pump.initial_status = status

        # Valves
        for valve_id, valve_data in self.config_file["valves"].items():
            status = np.random.choice(
                [1, 0], p=[self.valve_open_prob, 1.0 - self.valve_open_prob]
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
                        "res_head_hi"
                    ],
                    high=self.config_file["reservoirs"][reservoir_id][
                        "res_head_lo"
                    ],
                )

                reservoir.head_timeseries.base_value = random_head

                output_snpsht_values[reservoir_id] = {}
                output_snpsht_values[reservoir_id][
                    "input_base_head"
                ] = random_head
                reservoir.head_timeseries.pattern_name = None

        # Run simulation
        sim = wntr.sim.EpanetSimulator(self.wn)
        results_input = sim.run_sim()

        return results_input, output_snpsht_values

    def generate_output_snapshot(self, input_values):
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
                p=[self.pipe_open_prob, 1.0 - self.pipe_open_prob],
            )
            pipe = self.wn.get_link(pipe_id)
            pipe.initial_status = status

        # Pumps
        for pump_id, pump_data in self.config_file["pumps"].items():
            status = np.random.choice(
                [1, 0], p=[self.pump_open_prob, 1.0 - self.pump_open_prob]
            )
            pump = self.wn.get_link(pump_id)
            pump.initial_status = status

        # Valves
        for valve_id, valve_data in self.config_file["valves"].items():
            status = np.random.choice(
                [1, 0],
                p=[self.valve_open_prob, 1.0 - self.valve_open_prob],
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

    def _generate_one_sample(self):
        results_input, input_values = self.generate_input_snapshot()
        results_output = self.generate_output_snapshot(input_values)

        # Gather Node features for single snapshot
        input_node_features_values = []
        output_node_features_values = []
        for node_name, node_idx in self.node_id_map.items():
            (
                tmp_input_node_features,
                tmp_output_node_features,
            ) = (
                [],
                [],
            )
            for input_node_feature in self.input_node_features:
                tmp_input_node_features.append(
                    results_input.node[input_node_feature][node_name].values[0]
                )
            input_node_features_values.append(tmp_input_node_features)

            for output_node_feature in self.output_node_features:
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
            (
                tmp_input_edge_features,
                tmp_output_edge_features,
            ) = (
                [],
                [],
            )
            for input_edge_feature in self.input_edge_features:
                tmp_input_edge_features.append(
                    results_input.link[input_edge_feature][link_name].values[0]
                )
            input_edge_features_values.append(tmp_input_edge_features)

            for output_edge_feature in self.output_edge_features:
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
        print(f"=== Generating {self.name} scenarios ===")

        start_time = time()
        for scene_i in tqdm(range(self.num_scenarios)):
            (
                input_node_features,
                output_node_features,
                input_edge_features,
                output_edge_features,
            ) = self._generate_one_sample()

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

        end_time = time()
        print(f"Elapsed time: {np.round(end_time - start_time, 2)}s")


if __name__ == "__main__":
    NAME = "Train"
    INP_FILE = "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net1.inp"
    CONFIG_FILE = "C:/Projects/Time Series Analysis/wdn-state-estim/src/gnn-pressure-forecasting/network_configs/Net1/raw_config.json"
    TS = 3600
    NUM_SCENARIOS = 1000
    INPUT_NODE_FEATS = ["pressure", "head"]
    INPUT_EDGE_FEATS = ["flowrate"]
    OUTPUT_NODE_FEATS = ["pressure"]
    OUTPUT_EDGE_FEATS = ["flowrate"]

    scenegen = SceneGenerator(
        name=NAME,
        inp_file=INP_FILE,
        config_file=CONFIG_FILE,
        timestep=TS,
        num_scenarios=NUM_SCENARIOS,
        pipe_open_prob=1,
        pump_open_prob=1,
        valve_open_prob=1,
        remove_controls=True,
        input_node_features=INPUT_NODE_FEATS,
        output_node_features=OUTPUT_NODE_FEATS,
        input_edge_features=INPUT_EDGE_FEATS,
        output_edge_features=OUTPUT_EDGE_FEATS,
        random_seed=42,
    )
    scenegen.generate_scenarios()
