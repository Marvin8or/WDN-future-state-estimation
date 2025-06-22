# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:14:42 2025

@author: gabri
"""
import numpy as np
import wntr
from wntr.metrics import expected_demand

wn = wntr.network.WaterNetworkModel(
    "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net1.inp"
)


# Calculate theoretical maximum demand based on pipe capacities
# max_feasible_demand = expected_demand(wn)
# for junction in wn.junctions():
#     assigned_demand = junction.demand_timeseries_list[0].base_value
#     if assigned_demand > max_feasible_demand.loc[junction.name]:
#         print(
#             f"Demand exceeds feasible capacity at {junction.name}: {assigned_demand:.2f} > {max_feasible_demand.loc[junction.name]:.2f}"
#         )
# Calculate maximum theoretical flow to each node
def distance(x1, x2, y1, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


source_capacity = sum(reservoir.head for reservoir in wn.reservoirs())
for junction in wn.junctions():
    # Estimate pipe capacity to node
    connected_pipes = 
    for pipe in 
    path_capacity = min(
        pipe.diameter**2.63 for pipe in topological_distance(wn, junction)
    )  # Hazen-Williams approximation
    max_feasible_demand = (
        0.278 * path_capacity * source_capacity**0.54
    )  # Simplified estimate

    if junction.demand_timeseries_list[0].base_value > max_feasible_demand:
        print(
            f"Demand exceeds feasible capacity at {junction.name}: {sampled_demand} > {max_feasible_demand:.2f}"
        )
