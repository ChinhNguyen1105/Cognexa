from pathlib import Path

from cognexa.services.neural_classifier import (
    NeuralDocumentClassifier,
)


DATASET = Path(
    "data/samples/datasets/documents.csv"
)


def test_neural_document_classifier_end_to_end(
    tmp_path,
):
    from cognexa.ml.dataset import load_dataset

    texts, labels = load_dataset(DATASET)

    model_path = (
        tmp_path
        / "document_classifier.pth"
    )

    classifier = NeuralDocumentClassifier(
        hidden_size=16,
        learning_rate=0.01,
    )

    history = classifier.train(
        texts=texts,
        labels=labels,
        epochs=50,
        batch_size=4,
    )

    assert len(history) == 50

    assert history[-1] < history[0]

    classifier.save(model_path)

    assert model_path.exists()

    loaded_classifier = (
        NeuralDocumentClassifier()
    )

    loaded_classifier.load(model_path)

    predictions = loaded_classifier.predict(
        [
            "machine learning neural networks"
        ]
    )

    assert predictions[0] == "ai"

    predictions = loaded_classifier.predict(
        [
            "python programming functions"
        ]
    )

    assert predictions[0] == "programming"