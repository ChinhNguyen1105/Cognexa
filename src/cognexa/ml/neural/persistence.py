from pathlib import Path

import torch
from torch import nn

from cognexa.ml.neural.features import NeuralFeatureExtractor


def save_model(
    model: nn.Module,
    feature_extractor: NeuralFeatureExtractor,
    path: str | Path,
) -> None:
    """Save neural model and feature extractor to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vectorizer": feature_extractor.extractor.vectorizer,
        },
        path,
    )


def load_model(
    model: nn.Module,
    feature_extractor: NeuralFeatureExtractor,
    path: str | Path,
) -> nn.Module:
    """Load neural model and feature extractor from disk."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    feature_extractor.extractor.vectorizer = (
        checkpoint["vectorizer"]
    )

    return model