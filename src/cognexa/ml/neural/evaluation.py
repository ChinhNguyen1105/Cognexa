import torch
from torch import nn


def evaluate_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Evaluate classification accuracy."""

    model.eval()

    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == labels).sum().item()

    return correct / len(labels)
