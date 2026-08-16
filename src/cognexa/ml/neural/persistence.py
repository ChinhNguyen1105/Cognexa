from pathlib import Path
from typing import Any

import torch
from torch import nn

from cognexa.ml.neural.features import NeuralFeatureExtractor


def save_model(
    model: nn.Module,
    feature_extractor: NeuralFeatureExtractor,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save neural model, vectorizer and metadata."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vectorizer": (
                feature_extractor
                .extractor
                .vectorizer
            ),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
) -> dict[str, Any]:
    """Load a complete neural checkpoint."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def load_model(
    model: nn.Module,
    feature_extractor: NeuralFeatureExtractor,
    path: str | Path,
) -> nn.Module:
    """Load neural weights and TF-IDF vectorizer."""

    checkpoint = load_checkpoint(path)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    feature_extractor.extractor.vectorizer = (
        checkpoint["vectorizer"]
    )

    model.eval()

    return model