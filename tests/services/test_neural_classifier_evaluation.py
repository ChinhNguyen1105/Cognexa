
import torch

from cognexa.services.neural_classifier import NeuralDocumentClassifier


def test_neural_classifier_can_evaluate():
    texts = [
        "machine learning",
        "deep learning",
        "machine learning",
        "deep learning",
    ]

    labels = torch.tensor([0, 1, 0, 1])

    classifier = NeuralDocumentClassifier(
        hidden_size=4,
        num_classes=2,
        learning_rate=0.01,
    )

    classifier.train(
        texts=texts,
        labels=labels,
        epochs=10,
        batch_size=2,
    )

    accuracy = classifier.evaluate(
        texts=texts,
        labels=labels,
    )

    assert 0.0 <= accuracy <= 1.0
