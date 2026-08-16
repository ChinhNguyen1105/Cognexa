from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cognexa.ml.neural.dataset import DocumentDataset
from cognexa.ml.neural.evaluation import evaluate_accuracy
from cognexa.ml.neural.features import NeuralFeatureExtractor
from cognexa.ml.neural.inference.predictor import NeuralPredictor
from cognexa.ml.neural.models.document_classifier import DocumentClassifier
from cognexa.ml.neural.persistence import load_model, save_model
from cognexa.ml.neural.training.trainer import NeuralTrainer


class NeuralDocumentClassifier:
    """High-level service for neural document classification."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        learning_rate: float = 0.001,
    ):
        self.feature_extractor = NeuralFeatureExtractor()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.model = None
        self.trainer = None
        self.predictor = None

    def _build_model(self, input_size: int) -> None:
        self.model = DocumentClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
        )

        self.trainer = NeuralTrainer(
            model=self.model,
            learning_rate=self.learning_rate,
        )

        self.predictor = NeuralPredictor(self.model)

    def train(
        self,
        texts: list[str],
        labels: torch.Tensor,
        epochs: int,
        batch_size: int = 32,
    ) -> list[float]:

        features = self.feature_extractor.fit_transform(texts)

        self._build_model(
            input_size=features.shape[1],
        )

        dataset = DocumentDataset(
            features,
            labels,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        return self.trainer.fit(
            loader,
            epochs=epochs,
        )

    def predict(self, texts: list[str]) -> torch.Tensor:
        features = self.feature_extractor.transform(texts)

        return self.predictor.predict(features)

    def evaluate(
        self,
        texts: list[str],
        labels: torch.Tensor,
    ) -> float:
        features = self.feature_extractor.transform(texts)

        return evaluate_accuracy(
            self.model,
            features,
            labels,
        )

    def save(self, path: str | Path) -> None:
        save_model(
            self.model,
            self.feature_extractor,
            path,
        )

    def load(self, path: str | Path) -> None:
        load_model(
            self.model,
            self.feature_extractor,
            path,
        )