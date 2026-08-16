
import torch
from torch.utils.data import DataLoader

from cognexa.ml.neural.dataset import DocumentDataset
from cognexa.ml.neural.models.document_classifier import DocumentClassifier
from cognexa.ml.neural.training.trainer import NeuralTrainer


def test_fit_returns_loss_history():
    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    dataset = DocumentDataset(features, labels)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    model = DocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    trainer = NeuralTrainer(
        model,
        learning_rate=0.01,
    )

    history = trainer.fit(
        loader,
        epochs=5,
    )

    assert len(history) == 5
    assert all(isinstance(loss, float) for loss in history)
    assert all(loss > 0 for loss in history)


def test_fit_reduces_loss():
    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    dataset = DocumentDataset(features, labels)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    model = DocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    trainer = NeuralTrainer(
        model,
        learning_rate=0.01,
    )

    history = trainer.fit(
        loader,
        epochs=100,
    )

    assert history[-1] < history[0]
