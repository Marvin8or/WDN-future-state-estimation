import argparse
import json
import os
from datetime import datetime
from dataset_generation.SceneGenerator import SceneGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate water distribution network scenarios"
    )

    # Required arguments
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--name", type=str, required=True, help="Name of the scenario set"
    )
    parser.add_argument(
        "--inp_file", type=str, required=True, help="Path to the INP file"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the configuration JSON file",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1000,
        help="Number of scenarios to generate",
    )
    parser.add_argument(
        "--pipe_open_prob",
        type=float,
        default=0.95,
        help="Probability of pipes being open",
    )
    parser.add_argument(
        "--pump_open_prob",
        type=float,
        default=0.95,
        help="Probability of pumps being open",
    )
    parser.add_argument(
        "--valve_open_prob",
        type=float,
        default=0.95,
        help="Probability of valves being open",
    )
    parser.add_argument(
        "--remove_controls",
        # action="store_true",
        default=True,
        type=str,
        help="Remove controls from the network",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Feature lists
    parser.add_argument(
        "--input_node_features",
        type=str,
        nargs="+",
        default=["pressure"],
        help="List of input node features",
    )
    parser.add_argument(
        "--output_node_features",
        type=str,
        nargs="+",
        default=["pressure"],
        help="List of output node features",
    )
    parser.add_argument(
        "--input_edge_features",
        type=str,
        nargs="+",
        default=["flowrate"],
        help="List of input edge features",
    )
    parser.add_argument(
        "--output_edge_features",
        type=str,
        nargs="+",
        default=["flowrate"],
        help="List of output edge features",
    )
    parser.add_argument(
        "--skip_elements",
        type=str,
        nargs="+",
        default=[],
        help="List of elements (nodes/links) to skip",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.base_path, "scenes", args.name)

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save parameters to JSON with timestamp
    params = vars(args)
    params["generation_timestamp"] = timestamp

    # Initialize and run SceneGenerator
    generator = SceneGenerator(
        base_path=args.base_path,
        name=args.name,
        inp_file=args.inp_file,
        config_file=args.config_file,
        num_scenarios=args.num_scenarios,
        pipe_open_prob=args.pipe_open_prob,
        pump_open_prob=args.pump_open_prob,
        valve_open_prob=args.valve_open_prob,
        remove_controls=args.remove_controls,
        input_node_features=args.input_node_features,
        output_node_features=args.output_node_features,
        input_edge_features=args.input_edge_features,
        output_edge_features=args.output_edge_features,
        random_seed=args.random_seed,
    )

    # Generate scenarios
    generator.generate_scenarios()

    with open(
        os.path.join(output_dir, f"scene_generation_parameters.json"),
        "w",
    ) as f:
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    main()
