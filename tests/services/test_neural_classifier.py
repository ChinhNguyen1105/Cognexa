import torch

from cognexa.services.neural_classifier import NeuralDocumentClassifier


def test_neural_document_classifier_can_train_and_predict():
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

    history = classifier.train(
        texts=texts,
        labels=labels,
        epochs=10,
        batch_size=2,
    )

    assert len(history) == 10

    predictions = classifier.predict(
        ["machine learning"]
    )

    assert predictions.shape == (1,)
    assert predictions[0].item() in [0, 1]