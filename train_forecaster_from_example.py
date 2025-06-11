# -*- coding: utf-8 -*-
"""
Created on Thu May 29 09:40:13 2025

@author: gabri
"""
import os
import torch
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent.attentiontemporalgcn import A3TGCN2
from dataset import wntr_simulation_to_SGTF
from models import TGCNForecast, ConvGRUForecaster
import matplotlib.pyplot as plt


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, horizon, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(
            in_channels=node_features,
            out_channels=32,
            periods=periods,
            batch_size=batch_size,
        )  # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, horizon)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(
            x, edge_index
        )  # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h)
        h = self.linear(h)
        return h


if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    INP_FILE = "C:/Projects/Time Series Analysis/wdn-state-estim/src/networks/Net1.inp"
    TS = 10 * 60
    SIM_DUR = 10 * 24 * 3600
    BATCH_SIZE = 16
    NUM_EPOCH = 20
    SHUFFLE = True

    NODE_FEATURES = 1
    NUM_NODES = 11

    # Create dataset
    TRAIN_SPLIT_IDX = int(0.6 * SIM_DUR / TS)
    VAL_SPLIT_IDX = int(0.8 * SIM_DUR / TS)
    INPUT_SIZE = 18
    OUTPUT_SIZE = 6

    train_dataset, validation_dataset, test_dataset = wntr_simulation_to_SGTF(
        INP_FILE,
        TS,
        SIM_DUR,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        train_split_idx=TRAIN_SPLIT_IDX,
        val_split_idx=VAL_SPLIT_IDX,
    )

    train_x_tensor, val_x_tensor, test_x_tensor = (
        # Train
        torch.from_numpy(train_dataset.features)
        .type(torch.FloatTensor)
        .to(DEVICE),
        # Validation
        torch.from_numpy(validation_dataset.features)
        .type(torch.FloatTensor)
        .to(DEVICE),
        # Test
        torch.from_numpy(test_dataset.features)
        .type(torch.FloatTensor)
        .to(DEVICE),
    )
    train_target_tensor, val_target_tensor, test_target_tensor = (
        # Train
        torch.from_numpy(train_dataset.targets)
        .type(torch.FloatTensor)
        .to(DEVICE),
        # Validation
        torch.from_numpy(validation_dataset.targets)
        .type(torch.FloatTensor)
        .to(DEVICE),
        # Test
        torch.from_numpy(test_dataset.targets)
        .type(torch.FloatTensor)
        .to(DEVICE),
    )

    train_dataset_new, val_dataset_new, test_dataset_new = (
        torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor),
        torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor),
        torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor),
    )
    train_loader, val_loader, test_loader = (
        # Train
        torch.utils.data.DataLoader(
            train_dataset_new,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            drop_last=True,
            # pin_memory=True,
            # num_workers=4,
        ),
        # Validation
        torch.utils.data.DataLoader(
            val_dataset_new,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            drop_last=True,
            # pin_memory=True,
            # num_workers=4,
        ),
        # Test
        torch.utils.data.DataLoader(
            test_dataset_new,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            # pin_memory=True,
            # num_workers=4,
        ),
    )

    # Check where is the data stored
    for encoder_inputs, labels in train_loader:
        print(f"encoder_inputs device: {encoder_inputs.device}")
        print(f"labels device: {labels.device}")
        break

    print(f"Train inputs: {train_dataset.features.shape}")
    print(f"Train outputs: {train_dataset.targets.shape}")
    # %% Model Training
    model = TemporalGNN(
        node_features=NODE_FEATURES,
        periods=INPUT_SIZE,
        horizon=OUTPUT_SIZE,
        batch_size=BATCH_SIZE,
    ).to(DEVICE)
    print(f"model device is cuda: {next(model.parameters()).is_cuda}")

    print("\n=== Model Architecture ===")
    print(model)

    print("\n=== Model Layer Shapes ===")
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, "\t\t\t", model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print("Model's Total Params: ", total_param)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    print(
        " \n=== Optimizer's state_dict ==="
    )  # If you notice here the Attention is a trainable parameter
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for snapshot in train_dataset:
        static_edge_index = snapshot.edge_index.to(DEVICE)
        break

    print("\nStatic Graph Loading => DONE!")

    print("\n=== Starting Model Training ===")

    train_val_loss = {"train": [], "val": []}
    for epoch in range(NUM_EPOCH):
        # Model Training
        model.train()
        train_loss = 0
        running_train_loss = 0

        for encoder_inputs, labels in tqdm(
            train_loader, desc=f"EPOCH {epoch + 1}/{NUM_EPOCH}"
        ):
            optimizer.zero_grad()
            y_hat = model(encoder_inputs, static_edge_index)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)

        # Model Validation
        model.eval()
        val_loss = 0
        running_val_loss = 0

        with torch.no_grad():
            for encoder_inputs, labels in val_loader:
                y_hat = model(encoder_inputs, static_edge_index)
                loss = loss_fn(y_hat, labels)
                running_val_loss += loss.item()
            val_loss = running_val_loss / len(val_loader)

        # del y_hat, encoder_inputs, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_val_loss["train"].append(train_loss)
        train_val_loss["val"].append(val_loss)

        print(f"\n\tTrain loss: {train_loss} | Val loss: {val_loss}")

    print("\n=== Starting Model Testing ===")

    model.eval()
    with torch.no_grad():
        test_loss = 0
        running_test_loss = 0
        for encoder_inputs, labels in tqdm(
            test_loader, desc="Testing Progress"
        ):
            y_hat = model(encoder_inputs, static_edge_index)
            loss = loss_fn(y_hat, labels)
            running_test_loss += loss.item()
        test_loss = running_test_loss / len(test_loader)
        # test_loss.append(loss.item())

        # del encoder_inputs, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Average Test Loss: {test_loss}")

        print("\n=== Saving Model ===")
        year, month, day, hour, minute, second = (
            datetime.now().year,
            datetime.now().month,
            datetime.now().day,
            datetime.now().hour,
            datetime.now().minute,
            datetime.now().second,
        )

        path = f"models/Y{year}_M{month}_D{day}_h{hour}_m{minute}_s{second}"
        os.mkdir(path)
        torch.save(
            model.state_dict(),
            path + "/model.pkl",
        )
        plt.figure()
        plt.plot(train_val_loss["train"], color="blue", label="Train loss")
        plt.plot(train_val_loss["val"], color="red", label="Validation loss")
        plt.legend()
        plt.savefig(path + "/train_val_loss.jpeg")

    # %% Plot
    BATCH_IDX = 0
    NODE_ID_IDX = 0
    timestep = 0

    predictions = []
    ground_truth = []
    for idx in range(3):
        # print(y_hat[idx, NODE_ID_IDX, :].cpu().numpy().shape)
        plt.figure()
        x = np.arange(idx, idx + OUTPUT_SIZE)
        plt.plot(
            x,
            y_hat[idx, NODE_ID_IDX, :].cpu().numpy(),
            color="red",
            label="Predictions",
        )
        print(labels[idx, NODE_ID_IDX, :].cpu().numpy())
        plt.plot(
            x,
            labels[idx, NODE_ID_IDX, :].cpu().numpy(),
            color="green",
            label="Ground Truth",
        )
        plt.plot(x, labels[idx, NODE_ID_IDX, :].cpu().numpy(), "go")
        plt.legend()
        plt.show()
