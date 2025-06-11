# -*- coding: utf-8 -*-
"""
Graph matching GNN. Use a state of the network as input and output the next state of the WDN.
Created on Thu Jun  5 11:23:53 2025

@author: gabri
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset import generate_next_state_dataset
import pickle as pkl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class NextStatePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.2)
        x = F.relu(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    INP_FILE = "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net1.inp"
    TS = 10 * 60
    BATCH_SIZE = 32
    NUM_EPOCH = 50
    SHUFFLE = True

    TRAIN_NUM_SAMPLES = 1000
    TRAIN_SPLIT_PERC = 0.8
    TEST_NUM_SAMPLES = TS * 6

    NODE_FEATURES = 3

    if False:
        # TODO to device
        (
            train_dataset,
            validation_dataset,
            test_dataset,
        ) = generate_next_state_dataset(
            inp_file=INP_FILE,
            num_samples=TRAIN_NUM_SAMPLES,
            timestep=TS,
            pipe_open_perc=1,
            pump_open_perc=1,
            random_seed=42,
            train_split=TRAIN_SPLIT_PERC,
            test_size=TEST_NUM_SAMPLES,
            batch_size=BATCH_SIZE,
        )

        # Check where is the data stored
        for sample in train_dataset:
            inputs = sample.x
            outputs = sample.y
            print(f"\ninputs device: {inputs.device}")
            print(f"outputs device: {outputs.device}")
            break

        print(f"Train inputs: {train_dataset.dataset[0].x.shape}")
        print(f"Train outputs: {train_dataset.dataset[0].y.shape}")

        print("=== Saving Generated Dataset ===")
        with open(f"train_dataset_{TRAIN_NUM_SAMPLES}.pkl", "wb") as f:
            pkl.dump(train_dataset, f)
        with open(f"val_dataset_{TRAIN_NUM_SAMPLES}.pkl", "wb") as f:
            pkl.dump(validation_dataset, f)
        with open(f"test_dataset_{TRAIN_NUM_SAMPLES}.pkl", "wb") as f:
            pkl.dump(test_dataset, f)

    # %% Model Training

    with open(f"train_dataset_{TRAIN_NUM_SAMPLES}.pkl", "rb") as f:
        train_dataset = pkl.load(f)
    with open(f"val_dataset_{TRAIN_NUM_SAMPLES}.pkl", "rb") as f:
        validation_dataset = pkl.load(f)
    with open(f"test_dataset_{TRAIN_NUM_SAMPLES}.pkl", "rb") as f:
        test_dataset = pkl.load(f)

    model = NextStatePredictor(NODE_FEATURES, 64, 1)

    print(f"model device is cuda: {next(model.parameters()).is_cuda}")

    print("\n=== Model Architecture ===")
    print(model)

    print("\n=== Model Layer Shapes ===")
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, "\t\t\t", model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print("Model's Total Params: ", total_param)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print(
        " \n=== Optimizer's state_dict ==="
    )  # If you notice here the Attention is a trainable parameter
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    print("\n=== Starting Model Training ===")

    train_val_loss = {"train": [], "val": []}
    for epoch in range(NUM_EPOCH):
        # Model Training
        model.train()
        train_loss = 0
        running_train_loss = 0

        for sample in tqdm(
            train_dataset, desc=f"EPOCH {epoch + 1}/{NUM_EPOCH}"
        ):
            optimizer.zero_grad()
            inputs, outputs, edge_index = sample.x, sample.y, sample.edge_index
            y_hat = model(inputs, edge_index)
            loss = loss_fn(y_hat, outputs)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_dataset)

        # Model Validation
        model.eval()
        val_loss = 0
        running_val_loss = 0

        with torch.no_grad():
            for sample in validation_dataset:
                inputs, outputs, edge_index = (
                    sample.x,
                    sample.y,
                    sample.edge_index,
                )
                y_hat = model(inputs, edge_index)
                loss = loss_fn(y_hat, outputs)
                running_val_loss += loss.item()
            val_loss = running_val_loss / len(validation_dataset)

        train_val_loss["train"].append(train_loss)
        train_val_loss["val"].append(val_loss)

        print(f"\n\tTrain loss: {train_loss} | Val loss: {val_loss}")

    print("\n=== Starting Model Testing ===")

    model.eval()
    with torch.no_grad():
        test_loss = 0
        running_test_loss = 0
        for sample in tqdm(test_dataset, desc="Testing Progress"):
            inputs, outputs, edge_index = (
                sample.x,
                sample.y,
                sample.edge_index,
            )
            y_hat = model(inputs, edge_index)
            loss = loss_fn(y_hat, outputs)
            running_test_loss += loss.item()
        test_loss = running_test_loss / len(test_dataset)

        print(f"\nAverage Test Loss: {test_loss}")

    plt.figure()
    plt.plot(train_val_loss["train"], color="blue", label="Train loss")
    plt.plot(train_val_loss["val"], color="red", label="Validation loss")
    plt.legend()
    plt.savefig("train_val_loss.jpeg")

    # %% Visualize
    ground_truth = []
    predictions = []
    num_batches = 500
    batch_ctr = 0
    for sample in test_dataset:
        y_hat = model(sample.x, sample.edge_index)

        gt_as_list = sample.to_data_list()
        for bi in range(sample.num_graphs):
            mask = sample.batch == bi
            y_hat_bi = y_hat[mask]
            predictions.append(y_hat_bi.detach().numpy())
            ground_truth.append(gt_as_list[bi].y.detach().numpy())

        batch_ctr += 1
        if batch_ctr == num_batches - 1:
            break

    ground_truth = np.hstack(ground_truth)
    predictions = np.hstack(predictions)

    node_id = 2
    plt.figure()
    plt.plot(ground_truth[node_id, :], color="blue", label="Ground Truth")
    plt.plot(predictions[node_id, :], color="red", label="Predictions")
    plt.plot(ground_truth[node_id, :], "bo")
    plt.plot(predictions[node_id, :], "ro")
    plt.legend()
    plt.show()
