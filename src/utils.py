import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.to(device))
            total_loss += criterion(outputs, targets.to(device)).item()
    return total_loss / len(dataloader)


def get_flat(model: nn.Module) -> np.ndarray:
    """Return 1-D numpy array with all parameters (no gradients needed)."""
    with torch.no_grad():
        return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()


def set_flat(model: nn.Module, vec: np.ndarray) -> None:
    """Write contents of 'vec' back into the network (in-place)."""
    vec = torch.from_numpy(vec)
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(vec[idx:idx + n].view_as(p).to(p.device))
            idx += n


def fitness(vec: np.ndarray, model, dataloader, criterion, device='cpu'):
    set_flat(model, vec)
    return evaluate(model, dataloader, criterion, device)


def evaluate_regression_metrics(model, dataloader, device='cpu'):
    was_training = model.training
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds.append(outputs.view(-1).cpu())
            targets.append(y.view(-1).cpu())

    if was_training:
        model.train()

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return {
        "mse": mean_squared_error(targets, preds),
        "mae": mean_absolute_error(targets, preds),
        "r2": r2_score(targets, preds),
    }
