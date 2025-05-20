import torch.nn as nn
import os, ast


class RetentionPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        INPUT_SIZE = int(os.getenv("INPUT_SIZE"))
        HIDDEN_SIZES = ast.literal_eval(os.getenv("HIDDEN_SIZES"))

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], 1)
        )

    def forward(self, x):
        return self.net(x)