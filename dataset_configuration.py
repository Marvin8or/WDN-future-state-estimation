# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:52:25 2025

@author: gabri
@description: Script used to generate the configuration of a network inp file.
"""
import os
import json
import argparse
import numpy as np
from datetime import datetime
from dataset.ConfigGenerator import ConfigGenerator


def generate_configuration(network_directory, network_name):
    """
    Generate summary files of a network configuration.

    Parameters
    ----------
    network_directory : str
        Path to the directory containing the network .inp file
    network_name : str
        Name of the network file (without .inp extension)
    """
    output_dir = "network_configs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        output_dir, f"{network_name}_summary_{timestamp}.txt"
    )

    with open(summary_file, "w") as f:
        f.write("Network Configuration Summary\n")
        f.write("=" * 50 + "\n\n")

    print(f"\n{'='*50}")
    print(f"Network: {network_name}")
    print(f"{'='*50}")

    # Initialize ConfigGenerator
    config_gen = ConfigGenerator(
        os.path.join(network_directory, f"{network_name}.inp")
    )

    # Create network-specific directory
    net_dir = os.path.join(output_dir, network_name)
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

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
        f.write(f"\nNetwork: {network_name}\n")
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
                f"{comp_type.capitalize()} Configuration for {network_name}\n"
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

    print(f"\nConfiguration files saved in: {net_dir}")
    print(f"\nSummary file saved as: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate configuration files for a water network model."
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

    args = parser.parse_args()

    generate_configuration(args.network_dir, args.network_name)
