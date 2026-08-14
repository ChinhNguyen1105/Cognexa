from pathlib import Path

from cognexa.ml.dataset import load_dataset


def test_load_dataset():
    path = Path("data/samples/datasets/documents.csv")

    texts, labels = load_dataset(path)

    assert len(texts) == 10
    assert len(labels) == 10
    assert texts[0] == "machine learning algorithms and models"
    assert labels[0] == "ai"