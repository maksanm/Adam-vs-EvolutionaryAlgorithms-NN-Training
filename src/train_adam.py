import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PeptideDataset
from model import RetentionPredictor
from utils import evaluate

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
        print(f'Epoch {epoch + 1:3d}/{ADAM_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')


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
    model = train_adam(model, criterion, train_dataloader, val_dataloader)
