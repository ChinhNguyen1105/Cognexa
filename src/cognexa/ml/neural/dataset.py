
import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    """PyTorch dataset for document classification."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length.")

        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
