from dotenv import load_dotenv
load_dotenv()

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PeptideDataset
from model import RetentionPredictor

# =================== HYPERPARAMETERS ===================
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

ADAM_LR = float(os.getenv("ADAM_LR"))
ADAM_EPOCHS = int(os.getenv("ADAM_EPOCHS"))

CMAES_GENERATIONS = int(os.getenv("CMAES_GENERATIONS"))
CMAES_SIGMA = float(os.getenv("CMAES_SIGMA"))
CMAES_ADAPT_INTERVAL = int(os.getenv("CMAES_ADAPT_INTERVAL"))


# =================== TRAINING FUNCTIONS ===================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
    return total_loss / len(dataloader)


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
        val_loss = evaluate(model, val_dataloader, criterion)
        print(f'Epoch {epoch + 1:3d}/{ADAM_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')


# GPT-generated
def train_cmaes_1_1(model, criterion, train_dataloader, val_dataloader):
    # Hyperparameters from the paper
    p_target = 0.25     # Target success rate
    c_p = 0.2           # Smoothing factor for success rate
    c_c = 0.25          # Evolution path decay rate
    c_1 = 0.1           # Covariance matrix learning rate
    d_sigma = 2.0       # Damping factor for sigma update
    p_thresh = 0.44     # Threshold for h
    min_var = 1e-8      # Minimum variance for margin correction

    # Initialize parameters
    current_params = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    best_loss = evaluate(model, train_dataloader, criterion)

    # Initialize covariance components
    C = torch.ones_like(current_params)  # Diagonal covariance
    A = torch.sqrt(C)                    # Cholesky factor
    sigma = torch.tensor(CMAES_SIGMA)
    p_c = torch.zeros_like(current_params)  # Evolution path
    p_succ = torch.tensor(p_target)       # Smoothed success rate

    for gen in range(CMAES_GENERATIONS):
        # 1. Generate candidate solution
        y_new = torch.randn_like(current_params)
        v_new = current_params + sigma * A * y_new

        # 2. Evaluate candidate on training set
        nn.utils.vector_to_parameters(v_new, model.parameters())
        candidate_loss = evaluate(model, train_dataloader, criterion)

        # 3. Update smoothed success rate
        success = float(candidate_loss < best_loss)
        p_succ = (1 - c_p) * p_succ + c_p * torch.tensor(success)

        # 4. Update step-size
        sigma_update = (p_succ - p_target) / (d_sigma * (1 - p_target))
        sigma *= torch.exp(sigma_update)

        if candidate_loss < best_loss:
            # 5. Accept successful candidate
            current_params = v_new.detach().clone()
            best_loss = candidate_loss

            # 6. Compute adaptation parameters
            h = 1.0 if p_succ < p_thresh else 0.0
            delta = (1 - h) * c_1 * c_c * (2 - c_c)

            # 7. Update evolution path
            p_c = (1 - c_c) * p_c + h * torch.sqrt(torch.tensor(c_c * (2 - c_c))) * y_new

            # 8. Update covariance matrix
            C = (1 - c_1 + delta) * C + c_1 * p_c**2

            # 9. Update Cholesky factor with margin correction
            A = torch.sqrt(torch.clamp(C, min=min_var))
        else:
            # Revert parameters if candidate rejected
            nn.utils.vector_to_parameters(current_params, model.parameters())

        # Calculate validation loss for logging
        val_loss = evaluate(model, val_dataloader, criterion)
        print(f'Gen {gen + 1:3d}/{CMAES_GENERATIONS} | Train Loss: {best_loss:.5f} | Val Loss: {val_loss:.5f} | σ: {sigma.item():.5f}')

    # Restore best parameters
    nn.utils.vector_to_parameters(current_params, model.parameters())
    return model


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
criterion = nn.MSELoss()

# =================== RUN EXPERIMENTS ===================
print("Training with Adam:")
train_adam(model, criterion, train_dataloader, val_dataloader)

print("\nTraining with CMA-ES 1+1:")
model = RetentionPredictor()  # Reset model
train_cmaes_1_1(model, criterion, train_dataloader, val_dataloader)
