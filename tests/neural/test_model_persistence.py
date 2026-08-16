import torch

from cognexa.ml.neural.features import NeuralFeatureExtractor
from cognexa.ml.neural.models.document_classifier import DocumentClassifier
from cognexa.ml.neural.persistence import load_model, save_model


def test_model_can_be_saved_and_loaded(tmp_path):
    model = DocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    feature_extractor = NeuralFeatureExtractor()

    feature_extractor.fit_transform(
        [
            "machine learning",
            "deep learning",
        ]
    )

    path = tmp_path / "document_classifier.pth"

    save_model(
        model,
        feature_extractor,
        path,
    )

    new_model = DocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    new_feature_extractor = NeuralFeatureExtractor()

    load_model(
        new_model,
        new_feature_extractor,
        path,
    )

    assert torch.equal(
        model(torch.tensor([[1.0, 0.0]])),
        new_model(torch.tensor([[1.0, 0.0]])),
    )

    assert (
        feature_extractor.extractor.vectorizer.vocabulary_
        == new_feature_extractor.extractor.vectorizer.vocabulary_
    )