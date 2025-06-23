import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PeptideDataset
from model import RetentionPredictor
from utils import evaluate, evaluate_regression_metrics

from dotenv import load_dotenv
load_dotenv(override=True)

# =================== HYPERPARAMETERS ===================
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

CMAES_GENERATIONS = int(os.getenv("CMAES_GENERATIONS"))
CMAES_INITIAL_SIGMA = float(os.getenv("CMAES_INITIAL_SIGMA"))
CMAES_P_TARGET = float(os.getenv("CMAES_P_TARGET"))
CMAES_C_P = float(os.getenv("CMAES_C_P"))
CMAES_C_COV = float(os.getenv("CMAES_C_COV"))
CMAES_D_SIGMA = float(os.getenv("CMAES_D_SIGMA"))
CMAES_P_THRESH = float(os.getenv("CMAES_P_THRESH"))

CMAES_SIGMA_MIN = float(os.getenv("CMAES_SIGMA_MIN"))
CMAES_EARLY_STOP_PATIENCE = int(os.getenv("CMAES_EARLY_STOP_PATIENCE"))

torch.manual_seed(int(os.getenv("RANDOM_SEED")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== (1+1)-CMA-ES with Colesky update procedures ======
def update_cholesky(A, z, c_cov):
    c_a = torch.sqrt(1 - torch.tensor(c_cov))
    z2 = torch.dot(z, z)

    factor = torch.sqrt(1 + ((1 - c_a**2) * z2) / (c_a**2)) - 1
    coeff = (c_a / z2) * factor
    Az = A @ z
    A_new = c_a * A + coeff * torch.ger(Az, z)
    return A_new


def update_step_size(sigma, p_succ, success, c_p, p_target, d_sigma):
    p_succ_new = (1 - c_p) * p_succ + c_p * success
    sigma *= torch.exp((p_succ_new - p_target) / (d_sigma * (1 - p_target)))
    return sigma, p_succ_new


# =================== TRAINING FUNCTION ===================
def train_cmaes_1_1(
    model,
    criterion,
    train_dataloader,
    val_dataloader,
    generations=CMAES_GENERATIONS,
    sigma_init=CMAES_INITIAL_SIGMA,
    p_target=CMAES_P_TARGET,
    c_p=CMAES_C_P,
    c_cov=CMAES_C_COV,
    d_sigma=CMAES_D_SIGMA,
    p_thresh=CMAES_P_THRESH,
    sigma_min=CMAES_SIGMA_MIN,
    early_stop_patience=CMAES_EARLY_STOP_PATIENCE
):
    model.to(DEVICE)

    current_params = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    best_loss = evaluate(model, train_dataloader, criterion, DEVICE)

    dim = current_params.shape[0]
    A = torch.eye(dim).to(DEVICE)
    sigma = torch.tensor(sigma_init).to(DEVICE)
    p_succ = torch.tensor(p_target)

    learning_history = {
        "generation": [],
        "train_loss": [],
        "val_loss": [],
        "eval_calls": 0,
        "sigma": [],
        "timestamp": []
    }

    start_time = time.time()
    no_improve_counter = 0

    for gen in range(generations):

        z = torch.randn(dim).to(DEVICE)
        y = A @ z

        # print(current_params.get_device())
        # print(sigma.get_device())
        # print(y.get_device())

        candidate_params = current_params + sigma * y

        nn.utils.vector_to_parameters(candidate_params, model.parameters())
        candidate_loss = evaluate(model, train_dataloader, criterion, DEVICE)
        learning_history["eval_calls"] += 1

        success = float(candidate_loss < best_loss)
        sigma, p_succ = update_step_size(sigma, p_succ, success, c_p, p_target, d_sigma)

        if success:
            current_params = candidate_params.detach().clone()
            best_loss = candidate_loss
            no_improve_counter = 0
            if p_succ < p_thresh:
                A = update_cholesky(A, z, c_cov)
        else:
            nn.utils.vector_to_parameters(current_params, model.parameters())
            no_improve_counter += 1

        val_loss = evaluate(model, val_dataloader, criterion, DEVICE)
        learning_history["timestamp"].append(time.time() - start_time)
        learning_history["eval_calls"] += 1

        learning_history["generation"].append(gen + 1)
        learning_history["train_loss"].append(best_loss)
        learning_history["val_loss"].append(val_loss)
        learning_history["sigma"].append(sigma.item())

        print(f'Gen {gen + 1:3d}/{CMAES_GENERATIONS} | Train Loss: {best_loss:.5f} | Val Loss: {val_loss:.5f} | σ: {sigma.item():.5f} | Success: {bool(success)}')

        if sigma.item() < sigma_min:
            print("Early stop: sigma ≈ 0")
            break

        if no_improve_counter >= early_stop_patience:
            print(f"Early stop: no improvement in {early_stop_patience} generations")
            break

    nn.utils.vector_to_parameters(current_params, model.parameters())
    final_metrics = evaluate_regression_metrics(model, val_dataloader, DEVICE)
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

    # =================== RUN EXPERIMENTS ===================
    print("\nTraining with CMA-ES 1+1:")
    model, history = train_cmaes_1_1(model, criterion, train_dataloader, val_dataloader)
    print("\n\nFinal Metrics Summary:")
    print('CMAES_C_COV', CMAES_C_COV)
    print('CMAES_D_SIGMA', CMAES_D_SIGMA)
    print("=" * 40)
    print(f"Val R2: {history['final_r2']:.3f} | MSE: {history['final_mse']:.4f} | MAE: {history['final_mae']:.4f} | Eval Calls: {history['eval_calls']} | Time: {history['timestamp'][-1]:.2f}s")
