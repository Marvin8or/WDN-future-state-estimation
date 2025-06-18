# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 11:00:09 2025

@author: gabri
@description: Script used to generate the configuration of a network inp file.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from dataset_generation.ConfigGenerator import ConfigGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate configuration files for a water network model."
    )
    parser.add_argument(
        "--dataset_configuration_name",
        type=str,
        required=True,
        help="Directory containing the configuration and the data for this dataset",
    )
    parser.add_argument(
        "--network_dir",
        type=str,
        required=True,
        help="Directory containing the network .inp file",
    )

    parser.add_argument(
        "--network_name",
        type=str,
        required=True,
        help="Name of the network file (without .inp extension)",
    )

    parser.add_argument(
        "--demand_is_quantile",
        type=bool,
        required=True,
    )

    parser.add_argument(
        "--diff_demand_is_quantile",
        type=bool,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = "datasets"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create network-specific directory
    net_dir = os.path.join(output_dir, args.dataset_configuration_name)
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    # Create a summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        net_dir,
        f"{args.network_name}_summary.txt",
    )

    with open(summary_file, "w") as f:
        f.write("Network Configuration Summary\n")
        f.write("=" * 50 + "\n\n")

    print(f"\n{'='*50}")
    print(f"Network: {args.network_name}")
    print(f"{'='*50}")

    # Initialize ConfigGenerator
    config_gen = ConfigGenerator(
        os.path.join(
            args.network_dir,
            f"{args.network_name}.inp",
        ),
        args.demand_is_quantile,
        args.diff_demand_is_quantile,
    )

    # Count components by type
    component_counts = {
        "junctions": 0,
        "reservoirs": 0,
        "tanks": 0,
        "pipes": 0,
        "pumps": 0,
        "valves": 0,
    }

    component_counts["junctions"] = len(config_gen.junctions_dict.keys())
    component_counts["reservoirs"] = len(config_gen.reservoirs_dict.keys())
    component_counts["tanks"] = len(config_gen.tanks_dict.keys())
    component_counts["pipes"] = len(config_gen.pipes_dict.keys())
    component_counts["pumps"] = len(config_gen.pumps_dict.keys())
    component_counts["valves"] = len(config_gen.valves_dict.keys())

    component_counts["total components"] = np.sum(
        [v for v in component_counts.values()]
    )

    # Save summary to the main summary file
    with open(summary_file, "a") as f:
        f.write(f"\nNetwork: {args.network_name}\n")
        f.write("-" * 30 + "\n")
        for comp_type, count in component_counts.items():
            f.write(f"{comp_type.capitalize()}: {count}\n")
        f.write("\n")

    # Save detailed component information
    for comp_type in [
        "junctions",
        "reservoirs",
        "tanks",
        "pipes",
        "pumps",
        "valves",
    ]:
        comp_file = os.path.join(net_dir, f"{comp_type}.txt")

        with open(comp_file, "w") as f:
            f.write(
                f"{comp_type.capitalize()} Configuration for {args.network_name}\n"
            )
            f.write("=" * 50 + "\n\n")

            # Write all components of this type
            for comp_key, comp_data in config_gen.config_dict[
                comp_type
            ].items():
                f.write(f"Component ID: {comp_key}\n")
                f.write("-" * 30 + "\n")
                for key, value in comp_data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

    # Save raw configuration data as JSON for potential programmatic use
    json_file = os.path.join(net_dir, "raw_config.json")
    with open(json_file, "w") as f:
        json.dump(config_gen.config_dict, f, indent=4)

    # Save generation parameters to JSON
    params = vars(args)
    params["generation_timestamp"] = timestamp

    params_file = os.path.join(net_dir, "config_generation_parameters.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

    print(f"\nConfiguration files saved in: {net_dir}")
    print(f"\nSummary file saved as: {summary_file}")
    print(f"\nParameters saved as: {params_file}")


if __name__ == "__main__":
    main()
