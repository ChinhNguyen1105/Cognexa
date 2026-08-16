import torch
from torch import nn


class DocumentClassifier(nn.Module):
    """Simple neural network for document classification."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
