from dotenv import load_dotenv
load_dotenv(override=True)

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PeptideDataset
from model import RetentionPredictor
from utils import fitness, get_flat, set_flat

# =================== HYPERPARAMETERS ===================
BATCH_SIZE          = int(os.getenv("BATCH_SIZE"))

DE_POP_SIZE         = int(os.getenv("DE_POP_SIZE"))
DE_F                = float(os.getenv("DE_F"))
DE_CR               = float(os.getenv("DE_CR"))
DE_GENERATIONS      = int(os.getenv("DE_GENERATIONS"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== TRAINING FUNCTION ===================
def train_de(model, criterion, train_dataloader, val_dataloader):
    """
    Differential-Evolution (DE/rand/1/bin) optimiser
    """
    model.to(DEVICE)
    dim   = get_flat(model).size
    print(f"Starting DE: {dim:,} parameters | pop={DE_POP_SIZE} "
          f"| F={DE_F} | CR={DE_CR} | generations={DE_GENERATIONS}")

    # Initial population
    base = get_flat(model)
    population = np.stack([base] +
                          [base + np.random.normal(0, 0.1, size=dim)
                           for _ in range(DE_POP_SIZE - 1)])
    scores = np.array([fitness(ind, model, train_dataloader, criterion)
                       for ind in population])

    best_idx   = int(np.argmin(scores))
    best_vec   = population[best_idx].copy()
    best_score = scores[best_idx]
    print(f"Initial best train loss = {best_score:.5f}")

    # Main loop
    for g in range(DE_GENERATIONS):
        for i in range(DE_POP_SIZE):
            # choose three distinct indices a, b, c (≠ i)
            idxs = list(range(DE_POP_SIZE))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            x_a, x_b, x_c = population[a], population[b], population[c]

            # 1) Mutation
            v = x_a + DE_F * (x_b - x_c)

            # 2) Binomial crossover
            cross = np.random.rand(dim) < DE_CR
            cross[random.randrange(dim)] = True
            trial = np.where(cross, v, population[i])

            # 3) Selection
            score_trial = fitness(trial, model, train_dataloader, criterion)
            if score_trial < scores[i]:
                population[i] = trial
                scores[i]     = score_trial

                if score_trial < best_score:
                    best_score = score_trial
                    best_vec   = trial.copy()
                    print(f"[gen {g:03d}] new best train loss = {best_score:.6f}")

        # Validation for overfitting control
        val_loss = fitness(best_vec, model, val_dataloader, criterion)
        print(f"Generation {g+1:3d}/{DE_GENERATIONS} | "
              f"Population mean: {scores.mean():.5f} | "
              f"Best train loss: {best_score:.5f} | "
              f"Val loss: {val_loss:.5f}")

    # Restore best parameters before returning
    set_flat(model, best_vec)
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
criterion = nn.SmoothL1Loss()

# =================== RUN EXPERIMENT ===================
print("\nTraining with Differential Evolution:")
model = train_de(model, criterion, train_dataloader, val_dataloader)