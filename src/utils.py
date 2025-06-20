import numpy as np
import torch
import torch.nn as nn


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
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


def fitness(vec: np.ndarray, model, dataloader, criterion):
    set_flat(model, vec)
    return evaluate(model, dataloader, criterion)
