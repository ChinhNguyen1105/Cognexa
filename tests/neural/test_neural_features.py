import torch

from cognexa.ml.neural.features import NeuralFeatureExtractor


def test_fit_transform_returns_tensor():
    extractor = NeuralFeatureExtractor()

    features = extractor.fit_transform(
        [
            "machine learning",
            "deep learning",
        ]
    )

    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.shape[0] == 2


def test_transform_returns_tensor():
    extractor = NeuralFeatureExtractor()

    extractor.fit_transform(
        [
            "machine learning",
            "deep learning",
        ]
    )

    features = extractor.transform(
        ["machine learning"]
    )

    assert isinstance(features, torch.Tensor)
    assert features.dtype == torch.float32
    assert features.shape[0] == 1