import torch

from cognexa.ml.classical.features import TextFeatureExtractor


class NeuralFeatureExtractor:
    """Convert text documents into PyTorch tensors."""

    def __init__(self):
        self.extractor = TextFeatureExtractor()

    def fit_transform(self, texts: list[str]) -> torch.Tensor:
        """Learn TF-IDF vocabulary and return PyTorch features."""
        features = self.extractor.fit_transform(texts)

        return torch.tensor(
            features.toarray(),
            dtype=torch.float32,
        )

    def transform(self, texts: list[str]) -> torch.Tensor:
        """Transform texts using the learned vocabulary."""
        features = self.extractor.transform(texts)

        return torch.tensor(
            features.toarray(),
            dtype=torch.float32,
        )