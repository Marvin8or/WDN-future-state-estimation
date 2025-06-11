# -*- coding: utf-8 -*-
"""
Created on Wed May 14 11:14:42 2025

@author: gabri
"""
import torch
import wntr
import numpy as np
import networkx as nx
import torch_geometric
from tqdm import tqdm
from time import time
from typing import List
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.signal.static_graph_temporal_signal import (
    StaticGraphTemporalSignal,
)


def collate_fn(dataset: Dataset):
    src_batch = Batch.from_data_list(dataset.source_graphs)
    tgt_batch = Batch.from_data_list(dataset.target_graphs)
    return src_batch, tgt_batch


def window_time_series(time_series, num_samples_window):
    dataset = []
    for ts_idx in range(len(time_series) - num_samples_window):
        window = time_series[ts_idx : ts_idx + num_samples_window]
        dataset.append(np.array(window))
    return np.vstack(dataset)


def generate_snapshot_dataset(
    wn, windowed_timestep_idx, simulation_results, input_size, output_size
) -> np.ndarray:
    input_dataset = []
    output_dataset = []
    for win_ts_idx in windowed_timestep_idx:
        node_features = []
        node_targets = []
        for node_idx, node_name in enumerate(wn.node_name_list):
            node_features.append(
                simulation_results.node["pressure"][node_name].values[
                    win_ts_idx
                ][:input_size]
            )
            node_targets.append(
                simulation_results.node["pressure"][node_name].values[
                    win_ts_idx
                ][-output_size:]
            )

        node_features = np.array(node_features).reshape(
            len(wn.node_name_list), 1, -1
        )
        node_targets = np.array(node_targets).reshape(
            len(wn.node_name_list), -1
        )
        input_dataset.append(node_features)
        output_dataset.append(node_targets)

    return np.array(input_dataset), np.array(output_dataset)


def wntr_sim_to_torch_dataset(
    inp_path,
    timestep,
    simulation_duration,
    input_size,
    output_size,
    validation_size,
    train_split,
    batch_size=1,
    shuffle=False,
):
    wn = wntr.network.WaterNetworkModel(inp_path)

    wn.options.time.duration = simulation_duration
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.pattern_timestep = timestep
    wn.options.time.report_timestep = timestep

    # Simulate hydraulics
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    G = wn.to_graph()
    node_id_map = {node: i for i, node in enumerate(G.nodes)}

    # Extract features
    timestep_idx = np.arange(
        0, int(simulation_duration / timestep) + 1, dtype=np.int32
    )
    timesteps = np.arange(0, simulation_duration + timestep, timestep)

    print(
        f"Creating dataset with {(simulation_duration//timestep)} graph snapshots!\nFirst {train_split} used for training!"
    )
    # Edge extraction need to be done only once since the edges are static
    edge_list = []
    for u, v in G.edges(data=False):
        ui, vi = node_id_map[u], node_id_map[v]
        edge_list.append([ui, vi])
        edge_list.append([vi, ui])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # ==========================
    # Generate Train dataset
    train_dataset = []
    train_timesteps = timesteps[:train_split]
    windowed_idx = window_time_series(
        timestep_idx[:train_split], input_size + output_size + validation_size
    )

    for widx in windowed_idx:
        node_features = []

        for node_idx, node_name in enumerate(wn.node_name_list):
            node_features.append(
                results.node["pressure"][node_name].values[widx]
            )

        node_features = np.vstack(node_features)
        x = torch.tensor(node_features, dtype=torch.float)
        train_dataset.append(
            Data(
                x=x,
                edge_index=edge_index,
                node_ids=node_id_map,
            )
        )
    train_dataset = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    train_dataset.num_node_features = input_size

    # ==========================
    # Generate Test dataset
    test_dataset = []
    test_timesteps = timesteps[train_split:]

    for widx in timestep_idx[train_split:]:
        node_features = []

        for node_idx, node_name in enumerate(wn.node_name_list):
            node_features.append(
                results.node["pressure"][node_name].values[widx]
            )

        node_features = np.vstack(node_features)
        x = torch.tensor(node_features, dtype=torch.float)
        test_dataset.append(
            Data(
                x=x,
                edge_index=edge_index,
                node_ids=node_id_map,
            )
        )
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=shuffle)
    test_dataset.num_node_features = input_size

    return (
        node_id_map,
        train_timesteps,
        train_dataset,
        test_timesteps,
        test_dataset,
    )


def wntr_simulation_to_SGTF(
    inp_path,
    timestep,
    simulation_duration,
    input_size,
    output_size,
    train_split_idx,
    val_split_idx,
):
    wn = wntr.network.WaterNetworkModel(inp_path)

    wn.options.time.duration = simulation_duration
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.pattern_timestep = timestep
    wn.options.time.report_timestep = timestep

    # Simulate hydraulics
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    G = wn.to_graph()
    node_id_map = {node: i for i, node in enumerate(G.nodes)}

    # Extract features
    timestep_idx = np.arange(
        0, int(simulation_duration / timestep) + 1, dtype=np.int32
    )
    timesteps = np.arange(0, simulation_duration + timestep, timestep)
    print(
        f"Creating dataset with {(simulation_duration//timestep)} graph snapshots!"
    )
    # Edge extraction need to be done only once since the edges are static
    edge_list = []
    for u, v in G.edges(data=False):
        ui, vi = node_id_map[u], node_id_map[v]
        edge_list.append([ui, vi])
        edge_list.append([vi, ui])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # train_timesteps = timesteps[:train_split_idx]
    train_timestep_idx = timestep_idx[:train_split_idx]
    train_windowed_idx = window_time_series(
        train_timestep_idx, input_size + output_size
    )

    # validation_timesteps = timesteps[
    #     train_split_idx : train_split_idx + val_split_idx
    # ]
    validation_timestep_idx = timestep_idx[train_split_idx:val_split_idx]
    val_windowed_idx = window_time_series(
        validation_timestep_idx, input_size + output_size
    )

    # test_timesteps = timesteps[train_split_idx + val_split_idx :]
    test_timestep_idx = timestep_idx[val_split_idx:]
    test_windowed_idx = window_time_series(
        test_timestep_idx, input_size + output_size
    )

    # Generate Snapshot Dataset
    train_input, train_targets = generate_snapshot_dataset(
        wn, train_windowed_idx, results, input_size, output_size
    )
    val_input, val_targets = generate_snapshot_dataset(
        wn, val_windowed_idx, results, input_size, output_size
    )
    test_input, test_targets = generate_snapshot_dataset(
        wn, test_windowed_idx, results, input_size, output_size
    )
    train_dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=None,
        features=train_input,
        targets=train_targets,
    )
    validation_dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=None,
        features=val_input,
        targets=val_targets,
    )
    test_dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=None,
        features=test_input,
        targets=test_targets,
    )
    return train_dataset, validation_dataset, test_dataset


def generate_config(inp_file):
    config_dir = {}
    wn = wntr.network.WaterNetworkModel(inp_file)
    for junction_name in wn.junction_name_list:
        config_dir[junction_name] = {}
        config_dir[junction_name]["type"] = "junction"

        junction = wn.get_node(junction_name)

        demand_pattern_id = junction.demand_pattern
        demand_pattern = wn.get_pattern(demand_pattern_id)

        config_dir[junction_name]["demand_hi"] = np.max(
            junction.base_demand * demand_pattern.multipliers
        )
        config_dir[junction_name]["demand_lo"] = np.min(
            junction.base_demand * demand_pattern.multipliers
        )
        config_dir[junction_name]["demand_diff_hi"] = np.max(
            np.diff(junction.base_demand * demand_pattern.multipliers)
        )
        config_dir[junction_name]["demand_diff_lo"] = np.min(
            np.diff(junction.base_demand * demand_pattern.multipliers)
        )
        config_dir[junction_name][
            "demand_pat_ts"
        ] = demand_pattern.time_options.pattern_timestep

    for reservoir_name in wn.reservoir_name_list:
        config_dir[reservoir_name] = {}
        config_dir[reservoir_name]["type"] = "reservoir"

        reservoir = wn.get_node(reservoir_name)

        config_dir[junction_name]["res_head_hi"] = None
        config_dir[junction_name]["res_head_lo"] = None
        config_dir[junction_name]["res_diff_head_hi"] = None
        config_dir[junction_name]["res_diff_head_lo"] = None

        if reservoir.head_timeseries.pattern_name is not None:
            config_dir[junction_name]["res_head_hi"] = np.max(
                reservoir.head_timeseries
            )
            config_dir[junction_name]["res_head_lo"] = np.min(
                reservoir.head_timeseries
            )
            config_dir[junction_name]["res_diff_head_hi"] = np.max(
                np.diff(reservoir.head_timeseries)
            )
            config_dir[junction_name]["res_diff_head_lo"] = np.min(
                np.diff(reservoir.head_timeseries)
            )

    for pump_name in wn.pump_name_list:
        config_dir[pump_name] = {}
        config_dir[pump_name]["type"] = "pump"

        pump = wn.get_link(pump_name)
        config_dir[pump_name]["pump_speed_hi"] = None
        config_dir[pump_name]["pump_speed_lo"] = None
        config_dir[pump_name]["pump_diff_speed_hi"] = None
        config_dir[pump_name]["pump_diff_speed_lo"] = None

        if pump.speed_timeseries.pattern_name is not None:
            config_dir[pump_name]["pump_speed_hi"] = np.max(
                pump.speed_timeseries
            )
            config_dir[pump_name]["pump_speed_lo"] = np.min(
                pump.speed_timeseries
            )
            config_dir[pump_name]["pump_diff_speed_hi"] = np.max(
                np.diff(pump.speed_timeseries)
            )
            config_dir[pump_name]["pump_diff_speed_lo"] = np.min(
                np.diff(pump.speed_timeseries)
            )

    return config_dir


def generate_next_state_dataset(
    inp_file,
    num_samples,
    timestep,
    pipe_open_perc,
    pump_open_perc,
    random_seed,
    train_split,
    test_size,
    batch_size,
):
    np.random.seed(random_seed)
    configuration = generate_config(inp_file)

    dataset = []
    print("=== Generating Train Dataset ===")
    start_time = time()
    for _ in tqdm(range(num_samples)):
        sample = generate_one_next_state_train_sample(
            inp_file, configuration, timestep, pipe_open_perc, pump_open_perc
        )
        # print(sample.x, sample.y)
        dataset.append(sample)
    print(f"\nElapsed time: {np.round(time() - start_time, 2)}")
    train_idx = int(num_samples * train_split)
    train_dataset = dataset[:train_idx]
    validation_dataset = dataset[train_idx:]
    test_dataset = generate_test_dataset(inp_file, timestep, test_size)

    return (
        DataLoader(train_dataset, batch_size=batch_size),
        DataLoader(validation_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
    )


def generate_test_dataset(inp_file, timestep, test_size):
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = timestep * test_size
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.report_timestep = timestep
    wn.options.hydraulic.demand_model = "DD"
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    G = wn.to_graph()
    node_id_map = {node: i for i, node in enumerate(G.nodes)}
    edge_list = []
    for u, v in G.edges(data=False):
        ui, vi = node_id_map[u], node_id_map[v]
        edge_list.append([ui, vi])
        edge_list.append([vi, ui])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    dataset = []
    indices = np.arange(test_size)

    input_indices, output_indices = indices[:-1], indices[1:]
    print("=== Generating Test Dataset ===")
    start_time = time()
    for input_idx, output_idx in zip(input_indices, output_indices):
        features, targets = [], []
        for node_id in wn.node_name_list:
            features.append(
                [
                    results.node["pressure"][node_id].values[input_idx],
                    results.node["head"][node_id].values[input_idx],
                    results.node["quality"][node_id].values[input_idx],
                ]
            )
            targets.append(
                [
                    results.node["pressure"][node_id].values[output_idx],
                ]
            )

        features, targets = np.vstack(features), np.vstack(targets)
        x, y = torch.tensor(features), torch.tensor(targets)
        dataset.append(Data(x=x, y=y, edge_index=edge_index))
    print(f"\nElapsed time: {np.round(time() - start_time, 2)}")
    return dataset


def generate_one_next_state_train_sample(
    inp_file, config_data, timestep, pipe_open_perc, pump_open_perc
):
    # Input to GNN
    for_output_sim = {}

    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = timestep
    wn.options.time.hydraulic_timestep = timestep
    wn.options.hydraulic.demand_model = "DD"

    for junction_name in wn.junction_name_list:
        junction = wn.get_node(junction_name)
        random_demand = np.random.uniform(
            low=config_data[junction_name]["demand_lo"],
            high=config_data[junction_name]["demand_hi"],
        )
        junction.demand_timeseries_list[0].base_value = random_demand
        for_output_sim[junction_name] = {}
        for_output_sim[junction_name]["input_demand"] = random_demand

        junction.demand_timeseries_list[0].pattern_name = None

    # There are no reservoirs in this network

    for pipe_name in wn.pipe_name_list:
        status = np.random.choice(
            ["OPEN", "CLOSED"], p=[pipe_open_perc, 1.0 - pipe_open_perc]
        )
        pipe = wn.get_link(pipe_name)
        pipe.initial_status = status

    for pump_name in wn.pump_name_list:
        status = np.random.choice(
            ["OPEN", "CLOSED"], p=[pump_open_perc, 1.0 - pump_open_perc]
        )
        pump = wn.get_link(pump_name)
        pump.initial_status = status

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results_input = sim.run_sim()

    # Output to GNN
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = timestep
    wn.options.time.hydraulic_timestep = timestep
    wn.options.hydraulic.demand_model = "DD"

    for junction_name in wn.junction_name_list:
        junction = wn.get_node(junction_name)

        demand_diff = np.random.uniform(
            low=config_data[junction_name]["demand_diff_lo"],
            high=config_data[junction_name]["demand_diff_hi"],
        )

        junction.demand_timeseries_list[0].base_value = (
            for_output_sim[junction_name]["input_demand"] + demand_diff
        )
        junction.demand_timeseries_list[0].pattern_name = None

    # There are no reservoirs in this network

    for pipe_name in wn.pipe_name_list:
        status = np.random.choice(
            ["OPEN", "CLOSED"], p=[pipe_open_perc, 1.0 - pipe_open_perc]
        )
        pipe = wn.get_link(pipe_name)
        pipe.initial_status = status

    for pump_name in wn.pump_name_list:
        status = np.random.choice(
            ["OPEN", "CLOSED"], p=[pump_open_perc, 1.0 - pump_open_perc]
        )
        pump = wn.get_link(pump_name)
        pump.initial_status = status

    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results_output = sim.run_sim()

    G = wn.to_graph()
    node_id_map = {node: i for i, node in enumerate(G.nodes)}
    edge_list = []
    for u, v in G.edges(data=False):
        ui, vi = node_id_map[u], node_id_map[v]
        edge_list.append([ui, vi])
        edge_list.append([vi, ui])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    features = []
    targets = []
    for node_idx, node_name in enumerate(wn.node_name_list):
        # TODO add other node features

        features.append(
            [
                results_input.node["pressure"][node_name].values[0],
                results_input.node["head"][node_name].values[0],
                results_input.node["quality"][node_name].values[0],
            ]
        )
        targets.append(
            [
                results_output.node["pressure"][node_name].values[0],
                # results_output.node["head"][node_name].values[0],
                # results_output.node["quality"][node_name].values[0],
            ]
        )

    features, targets = np.vstack(features), np.vstack(targets)
    x, y = torch.tensor(features, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )
    return Data(x=x, y=y, edge_index=edge_index)


def test_window_time_series():
    time_series = np.arange(1000)
    num_samples_window = 4

    dataset = window_time_series(time_series, num_samples_window)

    assert dataset.shape == (1000 - 4, 4), "Shape of dataset is not correct"

    for widx in range(dataset.shape[0]):
        ctr = 0
        ctr += widx
        for value in dataset[widx]:
            assert (
                ctr == value
            ), f"Value doesnt match counter value {ctr=}, {value=}"
            ctr += 1
    print("TEST: window_time_series -> OK")


if __name__ == "__main__":
    # test_window_time_series()
    INP_FILE = "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net1.inp"
    TS = 3600
    SIM_DUR = 24 * 3600
    # datasetNet1 = wntr_sim_to_torch_dataset(
    #     INP_FILE,
    #     TS,
    #     SIM_DUR,
    #     input_size=2,
    #     output_size=1,
    #     validation_size=2,
    #     batch_size=4,
    #     shuffle=False,
    # )
    """
    train, val, test = wntr_simulation_to_SGTF(
        INP_FILE,
        TS,
        SIM_DUR,
        input_size=6,
        output_size=6,
        train_split_idx=2000,
        val_split_idx=1000,
    )"""
    print()

    train_dataset, val_dataset, test_dataset = generate_next_state_dataset(
        inp_file=INP_FILE,
        num_samples=10,
        timestep=TS,
        pipe_open_perc=1,
        pump_open_perc=1,
        random_seed=42,
        train_split=0.7,
        test_size=10,
        batch_size=1,
    )
