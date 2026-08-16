import torch
from torch import nn

from cognexa.ml.neural.evaluation import evaluate_accuracy


class FixedModel(nn.Module):
    def forward(self, x):
        return torch.tensor([
            [3.0, 1.0],
            [1.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ])


def test_evaluate_accuracy():
    model = FixedModel()

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    accuracy = evaluate_accuracy(
        model,
        features,
        labels,
    )

    assert accuracy == 1.0
