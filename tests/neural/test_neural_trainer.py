
import torch
from torch.utils.data import DataLoader

from cognexa.ml.neural.dataset import DocumentDataset
from cognexa.ml.neural.models.document_classifier import DocumentClassifier
from cognexa.ml.neural.training.trainer import NeuralTrainer


def test_train_epoch_updates_model():
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

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss = trainer.train_epoch(loader)

    after = list(model.parameters())

    assert isinstance(loss, float)
    assert loss > 0

    assert any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter
        in zip(before, after)
    )
