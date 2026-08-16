
import torch
from torch import nn


class NeuralPredictor:
    """Run inference with a trained neural document classifier."""

    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features)
            predictions = torch.argmax(outputs, dim=1)

        return predictions
