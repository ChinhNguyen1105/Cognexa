
import torch
from torch.utils.data import DataLoader

from cognexa.ml.neural.dataset import DocumentDataset


def test_document_dataloader():
    features = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    dataset = DocumentDataset(features, labels)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batches = list(loader)

    assert len(batches) == 2

    features_batch, labels_batch = batches[0]

    assert features_batch.shape == (2, 3)
    assert labels_batch.shape == (2,)

    assert torch.equal(
        features_batch,
        torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]),
    )

    assert torch.equal(
        labels_batch,
        torch.tensor([0, 1]),
    )
