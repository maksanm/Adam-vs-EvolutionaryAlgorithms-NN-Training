import torch, os
from torch.utils.data import Dataset


class PeptideDataset(Dataset):
    def __init__(self, sequences, retention_times):
        self.sequences = sequences
        self.retention_times = retention_times
        self.AMINO_ACIDS = os.getenv("AMINO_ACIDS")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        counts = torch.zeros(len(self.AMINO_ACIDS))
        for aa in self.sequences[idx]:
            if aa in self.AMINO_ACIDS:
                counts[self.AMINO_ACIDS.index(aa)] += 1
        return counts.float(), torch.tensor([float(self.retention_times[idx])]).float()