# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:00:27 2025

@author: gabri
"""
import wntr
import numpy as np

class ConfigGenerator:
    """
    A class for generating configuration information from a water network model.
    This class extracts and processes various parameters from different network components
    (junctions, reservoirs, tanks) and stores them in a structured format.
    """

    def __init__(self, inp_file):
        """
        Initialize the ConfigGenerator with a water network model.

        Parameters
        ----------
        inp_file : str
            Path to the EPANET .inp file containing the water network model.
        """
        self.wn = wntr.network.WaterNetworkModel(inp_file)

        # Extract configuration information for each network component
        self.junction_config()
        self.reservoir_config()
        self.tank_config()
        self.pump_config()
        self.valve_config()
        self.pipe_config()

        self.config_dict = {
            "junctions": self.junctions_dict,
            "reservoirs": self.reservoirs_dict,
            "tanks": self.tanks_dict,
            "pipes": self.pipes_dict,
            "pumps": self.pumps_dict,
            "valves": self.valves_dict,
        }

    def junction_config(self):
        """
        Extract and process configuration information for all junctions in the network.

        For each junction, this method collects:
        - Node type
        - Elevation
        - Coordinates
        - Demand pattern information including:
          - Maximum and minimum demands
          - Maximum and minimum demand differences
          - Pattern timestep
        """
        self.junctions_dict = {}
        for junction_id in self.wn.junction_name_list:
            self.junctions_dict[junction_id] = {}

            junction = self.wn.get_node(junction_id)
            self.junctions_dict[junction_id]["type"] = junction.node_type
            self.junctions_dict[junction_id]["elevation"] = junction.elevation
            self.junctions_dict[junction_id][
                "corrdinates"
            ] = junction.coordinates

            demand_pattern_id = junction.demand_pattern
            demand_pattern = self.wn.get_pattern(demand_pattern_id)

            self.junctions_dict[junction_id]["junc_demand_hi"] = np.max(
                junction.base_demand * demand_pattern.multipliers
            )
            self.junctions_dict[junction_id]["junc_demand_lo"] = np.min(
                junction.base_demand * demand_pattern.multipliers
            )
            self.junctions_dict[junction_id]["junc_demand_diff_hi"] = np.max(
                np.diff(junction.base_demand * demand_pattern.multipliers)
            )
            self.junctions_dict[junction_id]["junc_demand_diff_lo"] = np.min(
                np.diff(junction.base_demand * demand_pattern.multipliers)
            )
            self.junctions_dict[junction_id][
                "junc_demand_pat_ts"
            ] = demand_pattern.time_options.pattern_timestep

    def reservoir_config(self):
        """
        Extract and process configuration information for all reservoirs in the network.

        For each reservoir, this method collects:
        - Node type
        - Base head (elevation)
        - Coordinates
        - Head timeseries statistics (if available):
          - Maximum and minimum head values
          - Maximum and minimum head differences
        """
        self.reservoirs_dict = {}
        for reservoir_id in self.wn.reservoir_name_list:
            reservoir = self.wn.get_node(reservoir_id)
            self.reservoirs_dict[reservoir_id] = {}
            self.reservoirs_dict[reservoir_id]["type"] = reservoir.node_type
            self.reservoirs_dict[reservoir_id][
                "base_head"
            ] = reservoir.base_head
            self.reservoirs_dict[reservoir_id][
                "corrdinates"
            ] = reservoir.coordinates

            self.reservoirs_dict[reservoir_id]["res_head_hi"] = None
            self.reservoirs_dict[reservoir_id]["res_head_lo"] = None
            self.reservoirs_dict[reservoir_id]["res_diff_head_hi"] = None
            self.reservoirs_dict[reservoir_id]["res_diff_head_lo"] = None

            if reservoir.head_timeseries.pattern_name is not None:
                self.reservoirs_dict[reservoir_id]["res_head_hi"] = np.max(
                    reservoir.head_timeseries
                )
                self.reservoirs_dict[reservoir_id]["res_head_lo"] = np.min(
                    reservoir.head_timeseries
                )
                self.reservoirs_dict[reservoir_id][
                    "res_diff_head_hi"
                ] = np.max(np.diff(reservoir.head_timeseries))
                self.reservoirs_dict[reservoir_id][
                    "res_diff_head_lo"
                ] = np.min(np.diff(reservoir.head_timeseries))

    def tank_config(self):
        """
        Extract and process configuration information for all tanks in the network.

        For each tank, this method collects:
        - Node type
        - Elevation
        - Coordinates
        - Initial level
        - Tank level limits (high and low)
        - Tank volume limits (high and low)
        - Diameter
        """
        self.tanks_dict = {}
        for tank_id in self.wn.tank_name_list:
            tank = self.wn.get_node(tank_id)
            self.tanks_dict[tank_id] = {}

            # Basic tank properties
            self.tanks_dict[tank_id]["type"] = tank.node_type
            self.tanks_dict[tank_id]["elevation"] = tank.elevation
            self.tanks_dict[tank_id]["coordinates"] = tank.coordinates

            # Tank level information
            self.tanks_dict[tank_id]["initial_level"] = tank.init_level
            self.tanks_dict[tank_id]["tank_level_hi"] = tank.min_level
            self.tanks_dict[tank_id]["tank_level_lo"] = tank.max_level

            # Tank geometry
            self.tanks_dict[tank_id]["diameter"] = tank.diameter
            self.tanks_dict[tank_id]["tank_volume_lo"] = tank.min_vol

    def pump_config(self):
        """
        Extract and process configuration information for all pumps in the network.

        For each pump, this method collects:
        - Pump type (power, head, or curve)
        - Start and end node information
        - Speed pattern statistics
        - Initial status
        - Efficiency
        """
        self.pumps_dict = {}
        for pump_id in self.wn.pump_name_list:
            pump = self.wn.get_link(pump_id)
            self.pumps_dict[pump_id] = {}

            # Basic pump properties
            self.pumps_dict[pump_id]["type"] = pump.link_type
            self.pumps_dict[pump_id]["start_node"] = pump.start_node_name
            self.pumps_dict[pump_id]["end_node"] = pump.end_node_name

            self.pumps_dict[pump_id]["speed_pattern_hi"] = None
            self.pumps_dict[pump_id]["speed_pattern_lo"] = None
            self.pumps_dict[pump_id]["speed_pattern_diff_hi"] = None
            self.pumps_dict[pump_id]["speed_pattern_diff_lo"] = None

            if pump.speed_timeseries.pattern_name is not None:
                self.pumps_dict[pump_id]["speed_pattern_hi"] = np.max(
                    pump.speed_timeseries
                )

                self.pumps_dict[pump_id]["speed_pattern_lo"] = np.min(
                    pump.speed_timeseries
                )
                self.pumps_dict[pump_id]["speed_pattern_diff_hi"] = np.max(
                    np.diff(pump.speed_timeseries)
                )
                self.pumps_dict[pump_id]["speed_pattern_diff_lo"] = np.min(
                    np.diff(pump.speed_timeseries)
                )

            # Status and operational parameters
            self.pumps_dict[pump_id]["initial_status"] = pump.initial_status
            self.pumps_dict[pump_id]["efficiency"] = pump.efficiency

    def valve_config(self):
        """
        Extract and process configuration information for all valves in the network.

        For each valve, this method collects:
        - Valve type (PRV, PSV, PBV, FCV, TCV, GPV)
        - Start and end node information
        - Initial status
        """
        self.valves_dict = {}
        for valve_id in self.wn.valve_name_list:
            valve = self.wn.get_link(valve_id)
            self.valves_dict[valve_id] = {}

            # Basic valve properties
            self.valves_dict[valve_id]["type"] = valve.link_type
            self.valves_dict[valve_id]["valve_type"] = valve.valve_type
            self.valves_dict[valve_id]["start_node"] = valve.start_node_name
            self.valves_dict[valve_id]["end_node"] = valve.end_node_name

            # Status information
            self.valves_dict[valve_id]["initial_status"] = valve.initial_status

    def pipe_config(self):
        """
        Extract and process configuration information for all pipes in the network.

        For each pipe, this method collects:
        - Pipe type
        - Start and end node information
        - Length
        - Diameter
        - Roughness coefficient
        - Minor loss coefficient
        - Initial status
        - Bulk reaction coefficient
        - Wall reaction coefficient
        """
        self.pipes_dict = {}
        for pipe_id in self.wn.pipe_name_list:
            pipe = self.wn.get_link(pipe_id)
            self.pipes_dict[pipe_id] = {}

            # Basic pipe properties
            self.pipes_dict[pipe_id]["type"] = pipe.link_type
            self.pipes_dict[pipe_id]["start_node"] = pipe.start_node_name
            self.pipes_dict[pipe_id]["end_node"] = pipe.end_node_name

            # Physical properties
            self.pipes_dict[pipe_id]["length"] = pipe.length
            self.pipes_dict[pipe_id]["diameter"] = pipe.diameter
            self.pipes_dict[pipe_id]["roughness"] = pipe.roughness
            self.pipes_dict[pipe_id]["minor_loss"] = pipe.minor_loss

            # Status information
            self.pipes_dict[pipe_id]["initial_status"] = pipe.initial_status

            # Water quality parameters
            self.pipes_dict[pipe_id]["bulk_coeff"] = pipe.bulk_coeff
            self.pipes_dict[pipe_id]["wall_coeff"] = pipe.wall_coeff
