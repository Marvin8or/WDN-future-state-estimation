# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:07:30 2025

@author: gabri
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp_file",
        type=str,
        required=True,
        help="Directory containing the network .inp file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Directory containing the network configuration file.",
    )
    parser.add_argument(
        "--ts",
        type=int,
        required=True,
        help="Timestep of the simulation, also the duration of the simulation.",
    )
    parser.add_argument(
        "--pipe_open_prob",
        type=float,
        default=1.0,
        help="Probability that a pipe will be open.",
    )
    parser.add_argument(
        "--pump_open_prob",
        type=float,
        default=1.0,
        help="Probability that a pump will be open.",
    )
    parser.add_argument(
        "--valve_open_prob",
        type=float,
        default=1.0,
        help="Probability that a valve will be open.",
    )
    parser.add_argument(
        "--remove_controls",
        type=bool,
        default=True,
        help="Remove all configured controls form the simulation.",
    )
