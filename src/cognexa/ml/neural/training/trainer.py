
import torch
from torch import nn
from torch.utils.data import DataLoader


class NeuralTrainer:
    """Train a PyTorch document classification model."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
    ):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()

        total_loss = 0.0

        for features, labels in loader:
            self.optimizer.zero_grad()

            predictions = self.model(features)

            loss = self.loss_fn(predictions, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def fit(
        self,
        loader: DataLoader,
        epochs: int,
    ) -> list[float]:
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        history = []

        for _ in range(epochs):
            loss = self.train_epoch(loader)
            history.append(loss)

        return history
