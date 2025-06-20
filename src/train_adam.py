import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PeptideDataset
from model import RetentionPredictor
from utils import evaluate, evaluate_regression_metrics

from dotenv import load_dotenv
load_dotenv(override=True)

# =================== HYPERPARAMETERS ===================
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

ADAM_LR = float(os.getenv("ADAM_LR"))
ADAM_EPOCHS = int(os.getenv("ADAM_EPOCHS"))

torch.manual_seed(int(os.getenv("RANDOM_SEED")))


# =================== TRAINING FUNCTION ===================
def train_adam(model, criterion, train_dataloader, val_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)

    learning_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "eval_calls": 0,
        "timestamp": []
    }

    start = time.time()
    for epoch in range(ADAM_EPOCHS):
        model.train()
        epoch_train_losses = []
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        val_loss = evaluate(model, val_dataloader, criterion, 'cpu')

        learning_history["epoch"].append(epoch + 1)
        learning_history["train_loss"].append(train_loss)
        learning_history["val_loss"].append(val_loss)
        learning_history["eval_calls"] += 1
        learning_history["timestamp"].append(time.time() - start)

        print(f'Epoch {epoch + 1:3d}/{ADAM_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')

    metrics = evaluate_regression_metrics(model, val_dataloader)
    learning_history["final_mse"] = metrics["mse"]
    learning_history["final_mae"] = metrics["mae"]
    learning_history["final_r2"] = metrics["r2"]

    return model, learning_history


if __name__ == "__main__":
    # =================== DATA PREPARATION ===================
    sequences = []
    retention_times = []
    with open(os.getenv("DATA_PATH")) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            seq, rt = parts
            sequences.append(seq)
            retention_times.append(float(rt))

    dataset = PeptideDataset(sequences, retention_times)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RetentionPredictor()
    criterion = nn.SmoothL1Loss()

    # =================== RUN EXPERIMENTS ===================
    print("Training with Adam:")
    model, history = train_adam(model, criterion, train_dataloader, val_dataloader)
