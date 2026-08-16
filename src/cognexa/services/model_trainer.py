from pathlib import Path

from cognexa.ml.dataset import load_dataset
from cognexa.ml.model import save_model
from cognexa.ml.classical.trainer import DocumentTrainer


def train_model(
    dataset_path: Path,
    model_path: Path,
) -> None:
    """Train and save the document classification model."""
    texts, labels = load_dataset(dataset_path)

    trainer = DocumentTrainer()
    trainer.train(texts, labels)

    save_model(trainer, model_path)