# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:33:20 2025

@author: gabri
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import wntr_sim_to_torch_dataset
from models import TGCNForecast, ConvGRUForecaster

if __name__ == "__main__":
    INP_FILE = "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net1.inp"
    TS = 10 * 60
    SIM_DUR = 2 * 24 * 3600
    BATCH_SIZE = 6
    NUM_EPOCH = 10
    NUM_NODES = 11

    # Create dataset
    train_size = int(0.7 * SIM_DUR / TS)
    input_size = 6
    output_size = 3
    validation_size = 3

    (
        node_id_map,
        train_timesteps,
        train_dataset,
        test_timesteps,
        test_dataset,
    ) = wntr_sim_to_torch_dataset(
        INP_FILE,
        timestep=TS,
        simulation_duration=SIM_DUR,
        input_size=input_size,
        output_size=output_size,
        validation_size=validation_size,
        train_split=train_size,
        batch_size=BATCH_SIZE,
    )

    # Define per batch tasks.
    # -------
    # Training
    # Set model to train mode
    # Set optimizer gradients to zero
    # Forward pass y_pred = model(inputs) x_s
    # Evaluate error/loss
    # Aggregate loss to epoch loss x_t
    # Compute epoch loss by dividing aggregated epoch loss to number of batches
    # Backpropagate
    # Advance optimizer step

    # Evaluation
    # Set model to evaluation mode
    # --------
    # Evaluate loss by rolling the window
    # Imput model lagged values used for training input = [......train_size - input_size: train_size] -> [0 ......]

    model = ConvGRUForecaster(
        in_channels=input_size, hidden_channels=12, out_channels=output_size
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-6
    )
    criterion = torch.nn.MSELoss()

    # %% Train model

    for epoch in range(NUM_EPOCH + 1):
        epoch_train_loss = 0
        epoch_val_loss = 0
        for batch in tqdm(train_dataset, desc=f"EPOCH: {epoch}"):
            optimizer.zero_grad()
            model.train()

            x = batch.x[:, :input_size]
            y = batch.x[:, input_size : input_size + output_size]

            edge_index = batch.edge_index
            y_pred = model(x, edge_index)
            loss = criterion(y_pred, y)
            epoch_train_loss += loss
            loss.backward()

            # Rolling validation forecast
            model.eval()
            val = batch.x[:, -validation_size:]
            val_y_pred = model(
                batch.x[:, -validation_size - input_size : -validation_size],
                edge_index,
            )
            val_loss = criterion(val_y_pred, val)
            epoch_val_loss += val_loss

        avg_epoch_train_loss = epoch_train_loss / len(train_dataset)
        avg_epoch_val_loss = epoch_val_loss / len(train_dataset)

        optimizer.step()

        print(
            f"Train loss {avg_epoch_train_loss:.3f} | Val loss: {avg_epoch_val_loss:.3f}"
        )
    if epoch == NUM_EPOCH:
        last_batch = batch.detach().clone()  # Used later for inference

    # %% Test model
    model.eval()
    with torch.no_grad():
        last_batch_graphs = last_batch.to_data_list()
        x = last_batch_graphs[-1].x[:, -input_size:]
        edge_index = last_batch_graphs[-1].edge_index
        forecasts = []
        test_data = []
        for step in test_dataset:
            # print(step)
            y_pred = model(x, edge_index)
            x = torch.cat((x, y_pred), dim=1)[:, -input_size:]
            forecasts.append(y_pred.detach().numpy())
            test_data.append(step.x.detach().numpy())

        forecasts = np.hstack(forecasts)
        test_data = np.hstack(test_data)

        # Nodewise metrics
        nodewise_results = {"MSE": {}}
        for node_id, node_idx in node_id_map.items():
            nodewise_results["MSE"][node_id] = criterion(
                torch.tensor(forecasts[node_idx, :]),
                torch.tensor(test_data[node_idx, :]),
            )
            print(
                f"Test loss node {node_id} {nodewise_results['MSE'][node_id]:.3f}"
            )

        test_loss = criterion(
            torch.tensor(forecasts),
            torch.tensor(test_data),
        )
        print(f"Test loss {test_loss:.3f}")

    # %% Plot values for a node
    NODE_ID = "10"
    node_idx = node_id_map[NODE_ID]

    plt.figure()
    plt.plot(
        test_timesteps, test_data[node_idx], label="test data", color="blue"
    )
    plt.plot(
        test_timesteps,
        forecasts[node_idx],
        label="Forcasted data",
        color="red",
    )
    plt.show()
    plt.legend()
