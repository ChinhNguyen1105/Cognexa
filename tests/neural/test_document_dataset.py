
import torch

from cognexa.ml.neural.dataset import DocumentDataset


def test_document_dataset():
    features = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])

    labels = torch.tensor([0, 1, 0])

    dataset = DocumentDataset(features, labels)

    assert len(dataset) == 3

    feature, label = dataset[1]

    assert torch.equal(feature, torch.tensor([0.4, 0.5, 0.6]))
    assert label.item() == 1
