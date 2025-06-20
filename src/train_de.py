import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PeptideDataset
from model import RetentionPredictor
from utils import fitness, get_flat, set_flat, evaluate_regression_metrics

from dotenv import load_dotenv
load_dotenv(override=True)

# =================== HYPERPARAMETERS ===================
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

DE_POP_SIZE = int(os.getenv("DE_POP_SIZE"))
DE_F = float(os.getenv("DE_F"))
DE_CR = float(os.getenv("DE_CR"))
DE_GENERATIONS = int(os.getenv("DE_GENERATIONS"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)


# =================== TRAINING FUNCTION ===================
def train_de(
    model,
    criterion,
    train_dataloader,
    val_dataloader,
    de_pop_size=DE_POP_SIZE,
    de_f=DE_F,
    de_cr=DE_CR,
    de_generations=DE_GENERATIONS
):
    model.to(DEVICE)
    dim = get_flat(model).size
    print(f"Starting DE: {dim:,} parameters | pop={de_pop_size} "
          f"| F={de_f} | CR={de_cr} | generations={de_generations}")

    base = get_flat(model)
    population = np.stack(
        [base] + [
            base + np.random.normal(0, 0.1, size=dim)
            for _ in range(de_pop_size - 1)
        ]
    )
    scores = np.array([fitness(ind, model, train_dataloader, criterion)
                       for ind in population])
    eval_calls = de_pop_size

    best_idx = int(np.argmin(scores))
    best_vec = population[best_idx].copy()
    best_score = scores[best_idx]
    print(f"Initial best train loss = {best_score:.5f}")

    learning_history = {
        "generation": [],
        "train_loss": [],
        "val_loss": [],
        "eval_calls": 0,
        "timestamp": []
    }

    start = time.time()

    for g in range(de_generations):
        for i in range(de_pop_size):
            idxs = list(range(de_pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            x_a, x_b, x_c = population[a], population[b], population[c]

            v = x_a + de_f * (x_b - x_c)

            cross = np.random.rand(dim) < de_cr
            cross[random.randrange(dim)] = True
            trial = np.where(cross, v, population[i])

            score_trial = fitness(trial, model, train_dataloader, criterion)
            eval_calls += 1
            if score_trial < scores[i]:
                population[i] = trial
                scores[i] = score_trial

                if score_trial < best_score:
                    best_score = score_trial
                    best_vec = trial.copy()
                    print(f"[gen {g:03d}] new best train loss = {best_score:.6f}")

        set_flat(model, best_vec)
        val_loss = fitness(best_vec, model, val_dataloader, criterion)
        eval_calls += 1

        learning_history["generation"].append(g + 1)
        learning_history["train_loss"].append(best_score)
        learning_history["val_loss"].append(val_loss)
        learning_history["eval_calls"] = eval_calls
        learning_history["timestamp"].append(time.time() - start)

        print(f"Generation {g + 1:3d}/{DE_GENERATIONS} | "
              f"Population mean: {scores.mean():.5f} | "
              f"Best train loss: {best_score:.5f} | "
              f"Val loss: {val_loss:.5f}")

    set_flat(model, best_vec)

    final_metrics = evaluate_regression_metrics(model, val_dataloader)
    learning_history["final_mse"] = final_metrics["mse"]
    learning_history["final_mae"] = final_metrics["mae"]
    learning_history["final_r2"] = final_metrics["r2"]
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

    # =================== RUN EXPERIMENT ===================
    print("\nTraining with Differential Evolution:")
    model, history = train_de(model, criterion, train_dataloader, val_dataloader)
