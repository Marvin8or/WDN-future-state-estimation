# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:20:09 2025

@author: gabri
Script to train Graph Matching GNN
"""
import os
import json
import torch
import logging
import inspect
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from dataset_generation.WDNDataset import WDNDataset
from models import model_configurations, optimizer_configurations


def get_arguments():
    parser = argparse.ArgumentParser(description="Train Graph Matching GNN")
    parser.add_argument(
        "--scenarios_path",
        type=str,
        required=True,
        help="Path to the generated scenarios directory",
    )
    parser.add_argument(
        "--num_train_scenarios",
        type=int,
        required=True,
        help="Number of training scenarios",
    )
    parser.add_argument(
        "--num_validation_scenarios",
        type=int,
        required=True,
        help="Number of validation scenarios",
    )
    parser.add_argument(
        "--num_test_scenarios",
        type=int,
        required=True,
        help="Number of test scenarios",
    )
    parser.add_argument(
        "--do_norm",
        type=bool,
        default=True,
        help="Whether to normalize the data",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="znorm",
        choices=["znorm", "minmax"],
        help="Normalization type",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        choices=["gcnconv_simple_16h"],
        help="Chose pre-configured model",
    )

    parser.add_argument(
        "--optim_config",
        type=str,
        required=True,
        choices=["adam_lr001"],
        help="Chose pre-configured optimizer",
    )

    parser.add_argument(
        "--loss_function",
        type=str,
        required=True,
        default=["MSE"],
        help="Loss function",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--keep_elements",
        nargs="+",
        default=[],
        help='Elements to keep (["all"] or list of node/link ids)',
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="node",
        choices=["node", "edge", "both"],
        help="Type of input data",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        default="node",
        choices=["node", "edge", "both"],
        help="Type of output data",
    )
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
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )

    parser.add_argument(
        "--save_experiment_base_path",
        type=str,
        required=True,
        default="experiments",
        help="Path where the experiment results are stored",
    )

    # Save the parsed argument in json file
    args = parser.parse_args()

    return args


def get_datasets(args, logger):
    train_dataset = WDNDataset(
        name="Train",
        logger=logger,
        scenarios_path=args.scenarios_path,
        num_scenarios=args.num_train_scenarios,
        mean=None,
        std=None,
        min=None,
        max=None,
        do_norm=args.do_norm,
        norm_type=args.norm_type,
        batch_size=args.batch_size,
        keep_elements=args.keep_elements,
        input_data=args.input_data,
        output_data=args.output_data,
        input_node_features=args.input_node_features,
        output_node_features=args.output_node_features,
        input_edge_features=args.input_edge_features,
        output_edge_features=args.output_edge_features,
    )

    validation_dataset = WDNDataset(
        name="Validation",
        logger=logger,
        scenarios_path=args.scenarios_path,
        num_scenarios=args.num_validation_scenarios,
        mean=train_dataset.mean,
        std=train_dataset.std,
        min=train_dataset.min,
        max=train_dataset.max,
        do_norm=args.do_norm,
        norm_type=args.norm_type,
        batch_size=args.batch_size,
        keep_elements=args.keep_elements,
        input_data=args.input_data,
        output_data=args.output_data,
        input_node_features=args.input_node_features,
        output_node_features=args.output_node_features,
        input_edge_features=args.input_edge_features,
        output_edge_features=args.output_edge_features,
    )

    test_dataset = WDNDataset(
        name="Test",
        logger=logger,
        scenarios_path=args.scenarios_path,
        num_scenarios=args.num_test_scenarios,
        mean=train_dataset.mean,
        std=train_dataset.std,
        min=train_dataset.min,
        max=train_dataset.max,
        do_norm=args.do_norm,
        norm_type=args.norm_type,
        batch_size=args.batch_size,
        keep_elements=args.keep_elements,
        input_data=args.input_data,
        output_data=args.output_data,
        input_node_features=args.input_node_features,
        output_node_features=args.output_node_features,
        input_edge_features=args.input_edge_features,
        output_edge_features=args.output_edge_features,
    )

    return (
        train_dataset.generate_DataLoader(),
        validation_dataset.generate_DataLoader(),
        test_dataset.generate_DataLoader(),
    )


def train_one_epoch(
    logger, model, optimizer, loss_func, train_dataset, epoch, num_epochs
):
    model.train()
    running_loss = 0
    for sample in tqdm(train_dataset, desc=f"EPOCH {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        (
            input_node_features,
            input_edge_features,
            output_features,
            edge_index,
        ) = (sample.x, sample.edge_attrs, sample.y, sample.edge_index)
        # logger.debug(f"edge_index type: {type(edge_index)}, {edge_index}")
        y_hat = model(input_node_features, input_edge_features, edge_index)
        loss = loss_func(y_hat, output_features)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return model, running_loss / len(train_dataset)


def validate_one_epoch(model, loss_func, validation_dataset):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for sample in validation_dataset:
            (
                input_node_features,
                input_edge_features,
                output_features,
                edge_index,
            ) = (sample.x, sample.edge_attrs, sample.y, sample.edge_index)
            y_hat = model(input_node_features, input_edge_features, edge_index)
            loss = loss_func(y_hat, output_features)
            running_loss += loss.item()
        return running_loss / len(validation_dataset)


def evaluate_model(model, loss_func, test_dataset):
    test_loss = validate_one_epoch(model, loss_func, test_dataset)
    return test_loss


def train(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = args.scenarios_path.split("/")[-1]

    experiment_folder = os.path.join(
        args.save_experiment_base_path,
        args.model_config + f"_{dataset_name}" + f"_{timestamp}",
    )

    os.makedirs(experiment_folder)

    logger_filename = os.path.join(experiment_folder, "resultslog.log")
    logging.basicConfig(
        filename=logger_filename,
        encoding="utf-8",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(timestamp)

    train_dataset, validation_dataset, test_dataset = get_datasets(
        args, logger
    )

    logger.info(f"Selected model: {args.model_config}")
    logger.info(f"Selected optimizer: {args.optim_config}")
    logger.info(f"Selected loss function: {args.loss_function}")

    logger.info("=" * 80)
    logger.info("Starting Model Training")
    logger.info("=" * 80)

    args, model = model_configurations.select_model(args)
    args, optimizer = optimizer_configurations.select_optimizer(
        args, model.parameters()
    )

    if args.loss_function == "MSE":
        loss_function = torch.nn.MSELoss()

    train_val_loss = defaultdict(list)
    for epoch in range(args.num_epochs):
        model, epoch_train_loss = train_one_epoch(
            logger,
            model,
            optimizer,
            loss_function,
            train_dataset,
            epoch,
            args.num_epochs,
        )
        epoch_val_loss = validate_one_epoch(
            model, loss_function, validation_dataset
        )
        train_val_loss["train"].append(epoch_train_loss)
        train_val_loss["val"].append(epoch_val_loss)

        print(
            f"\n\tTrain loss: {epoch_train_loss} | Val loss: {epoch_val_loss}"
        )
        # logger.info(
        #     f"EPOCH {epoch + 1}/{args.num_epochs}: Train loss: {epoch_train_loss} | Val loss: {epoch_val_loss}"
        # )

    logger.info("=" * 80)
    logger.info("Starting Model Testing")
    logger.info("=" * 80)
    test_loss = evaluate_model(model, loss_function, test_dataset)
    print(f"\nAverage Test Loss: {test_loss}")
    logger.info(f"Average Test Loss: {test_loss}")

    # if args.save_model:
    # Save to path (args.model_save_path)

    save_json_data_path = os.path.join(
        experiment_folder, "experiment_configuration_data.json"
    )
    params = vars(args)
    with open(
        save_json_data_path,
        "w",
    ) as jfile:
        json.dump(params, jfile, indent=4)

    logger.info(f"Saved json configuration at path: {save_json_data_path}")

    save_fig_path = os.path.join(experiment_folder, "train_val_loss.jpeg")
    plt.figure()
    plt.title("Training results for")
    plt.plot(train_val_loss["train"], color="blue", label="Train loss")
    plt.plot(train_val_loss["val"], color="red", label="Validation loss")
    plt.legend()
    plt.savefig(save_fig_path)

    logger.info(f"Saved training results figure at path: {save_fig_path}")


if __name__ == "__main__":
    args = get_arguments()
    train(args)
