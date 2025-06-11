# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 20:46:32 2025

@author: gabri
"""
import numpy as np
import wntr
import matplotlib.pyplot as plt

inp_file = (
    "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net6.inp"
)
node_id = "JUNCTION-0"

wn = wntr.network.WaterNetworkModel(inp_file)
junction = wn.get_node(node_id)
print(junction.demand_pattern)


print(junction.demand_timeseries_list)
pattern = wn.get_pattern(junction.demand_pattern)
print(junction.base_demand)
print(junction.base_demand * pattern.multipliers)

print(pattern.time_options)

print()

# %%
print("Reservoirs")
reservoir_name = wn.reservoir_name_list[0]
reservoir = wn.get_node(reservoir_name)
reservoir.head_timeseries.base_value = 0
print(reservoir.head_timeseries.base_value)
# print(reservoir.base_head)
# print(reservoir.head_pattern_name)
# print(reservoir.head_timeseries)
# print(reservoir.demand)
# print(reservoir.head)
# %%
tank_name = wn.tank_name_list
print("Tanks")
print(wn.tank_name_list)
print(tank_name[0])


pipes = wn.pipe_name_list
print(pipes)

pipe_name = "22"
pipe = wn.get_link(pipe_name)
print(pipe.status)

pumps_names = wn.pump_name_list
pump_name = pumps_names[0]
pump = wn.get_link(pump_name)
print(pump.speed_timeseries.pattern)

valves = wn.valve_name_list

print("get node")
print(dir(wn.get_node("9")))

"""
print(valves)
for junction_id in wn.junction_name_list:
    junction = wn.get_node(junction_id)
    print(f"Junction: {junction}")
    print(f"Base demand: {junction.base_demand}")
    demand_pattern_name = junction.demand_pattern
    pattern = wn.get_pattern(demand_pattern_name)
    print(f"Demand multipliers: {pattern.multipliers*junction.base_demand}")
    print(f"Demand pattern timestep: {pattern.time_options.pattern_timestep}")
"""
# %%
# Print node features
wn = wntr.network.WaterNetworkModel(inp_file)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
for node_id in wn.node_name_list:
    node = wn.get_node(node_id)
    print(f"\n=== Node: {node_id} ===")
    for feat in dir(node):
        if "_" not in feat:
            print(f"{feat}: {getattr(node, feat)}")

for link_id in wn.link_name_list:
    link = wn.get_link(link_id)
    print(f"\n=== Link: {link_id} ===")
    for feat in dir(link):
        if "_" not in feat:
            print(f"{feat}: {getattr(link, feat)}")
