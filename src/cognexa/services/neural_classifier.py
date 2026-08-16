from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cognexa.ml.neural.dataset import DocumentDataset
from cognexa.ml.neural.evaluation import evaluate_accuracy
from cognexa.ml.neural.features import NeuralFeatureExtractor
from cognexa.ml.neural.inference.predictor import NeuralPredictor
from cognexa.ml.neural.models.document_classifier import DocumentClassifier
from cognexa.ml.neural.persistence import (
    load_checkpoint,
    load_model,
    save_model,
)
from cognexa.ml.neural.training.trainer import NeuralTrainer


class NeuralDocumentClassifier:
    """High-level service for neural document classification."""

    def __init__(
        self,
        hidden_size: int = 32,
        num_classes: int | None = None,
        learning_rate: float = 0.001,
    ):
        self.feature_extractor = NeuralFeatureExtractor()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.num_classes = num_classes or 0
        self.input_size = 0

        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        self.model = None
        self.trainer = None
        self.predictor = None

    def _build_model(
        self,
        input_size: int,
        num_classes: int,
    ) -> None:
        """Create the neural network and related components."""

        self.input_size = input_size
        self.num_classes = num_classes

        self.model = DocumentClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_classes=num_classes,
        )

        self.trainer = NeuralTrainer(
            model=self.model,
            learning_rate=self.learning_rate,
        )

        self.predictor = NeuralPredictor(self.model)

    def _encode_labels(
        self,
        labels: list[str],
    ) -> torch.Tensor:
        """Convert string labels into integer class IDs."""

        unique_labels = sorted(set(labels))

        self.label_to_id = {
            label: index
            for index, label in enumerate(unique_labels)
        }

        self.id_to_label = {
            index: label
            for label, index in self.label_to_id.items()
        }

        encoded = [
            self.label_to_id[label]
            for label in labels
        ]

        return torch.tensor(
            encoded,
            dtype=torch.long,
        )

    def _set_numeric_label_mapping(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Handle labels that are already encoded as integers.

        Example:
            tensor([0, 1, 0, 1])
        """

        labels = labels.long()

        unique_labels = sorted(
            set(labels.tolist())
        )

        self.label_to_id = {
            str(label): int(label)
            for label in unique_labels
        }

        self.id_to_label = {
            int(label): str(label)
            for label in unique_labels
        }

        return labels

    def _decode_predictions(
        self,
        predictions: torch.Tensor,
    ) -> list[str]:
        """Convert class IDs back to string labels."""

        return [
            self.id_to_label[int(prediction)]
            for prediction in predictions
        ]

    def train(
        self,
        texts: list[str],
        labels,
        epochs: int,
        batch_size: int = 32,
    ) -> list[float]:
        """Train the neural document classifier."""

        if not texts:
            raise ValueError(
                "Training dataset is empty."
            )

        if len(texts) != len(labels):
            raise ValueError(
                "Texts and labels must have the same length."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        # Text → TF-IDF → Tensor
        features = (
            self.feature_extractor
            .fit_transform(texts)
        )

        # Handle both:
        #   list[str]
        #   torch.Tensor
        if isinstance(labels, torch.Tensor):
            encoded_labels = (
                self._set_numeric_label_mapping(
                    labels
                )
            )
        else:
            encoded_labels = (
                self._encode_labels(labels)
            )

        # If num_classes was not explicitly provided,
        # infer it from the labels.
        if self.num_classes == 0:
            self.num_classes = len(
                self.label_to_id
            )

        # Build model after knowing feature dimension
        self._build_model(
            input_size=features.shape[1],
            num_classes=self.num_classes,
        )

        # Tensor → Dataset
        dataset = DocumentDataset(
            features,
            encoded_labels,
        )

        # Dataset → DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Training
        return self.trainer.fit(
            loader,
            epochs=epochs,
        )

    def predict_ids(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """Return numeric class predictions."""

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained or loaded."
            )

        features = (
            self.feature_extractor
            .transform(texts)
        )

        return self.predictor.predict(
            features
        )

    def predict(
        self,
        texts: list[str],
    ) -> list[str]:
        """Predict human-readable document labels."""

        predictions = self.predict_ids(texts)

        return self._decode_predictions(
            predictions
        )

    def evaluate(
        self,
        texts: list[str],
        labels,
    ) -> float:
        """Evaluate classification accuracy."""

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained or loaded."
            )

        if len(texts) != len(labels):
            raise ValueError(
                "Texts and labels must have the same length."
            )

        features = (
            self.feature_extractor
            .transform(texts)
        )

        if isinstance(labels, torch.Tensor):
            encoded_labels = labels.long()
        else:
            encoded_labels = torch.tensor(
                [
                    self.label_to_id[label]
                    for label in labels
                ],
                dtype=torch.long,
            )

        return evaluate_accuracy(
            self.model,
            features,
            encoded_labels,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save model, vectorizer and metadata."""

        if self.model is None:
            raise RuntimeError(
                "Cannot save an untrained model."
            )

        save_model(
            model=self.model,
            feature_extractor=self.feature_extractor,
            path=path,
            metadata={
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_classes": self.num_classes,
                "learning_rate": self.learning_rate,
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
            },
        )

    def load(
        self,
        path: str | Path,
    ) -> None:
        """Load a complete trained model."""

        checkpoint = load_checkpoint(path)

        metadata = checkpoint["metadata"]

        self.input_size = metadata["input_size"]
        self.hidden_size = metadata["hidden_size"]
        self.num_classes = metadata["num_classes"]
        self.learning_rate = metadata["learning_rate"]

        self.label_to_id = metadata["label_to_id"]

        self.id_to_label = {
            int(key): value
            for key, value in metadata[
                "id_to_label"
            ].items()
        }

        self._build_model(
            input_size=self.input_size,
            num_classes=self.num_classes,
        )

        load_model(
            model=self.model,
            feature_extractor=self.feature_extractor,
            path=path,
        )

        self.model.eval()