from pathlib import Path

import torch

from cognexa.classifier import DocumentClassifier
from cognexa.persistence import load_model, save_model


def test_save_and_load_model(tmp_path: Path):
    model = DocumentClassifier()

    path = tmp_path / "classifier.pth"

    save_model(model, str(path))

    loaded_model = DocumentClassifier()

    load_model(loaded_model, str(path))

    for original, loaded in zip(
        model.parameters(),
        loaded_model.parameters(),
    ):
        assert torch.equal(original, loaded)