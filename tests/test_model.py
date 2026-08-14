from pathlib import Path

from cognexa.ml.classifier import DocumentClassifier
from cognexa.ml.features import TextFeatureExtractor
from cognexa.ml.model import load_model, save_model


def test_save_and_load_model(tmp_path: Path):
    texts = [
        "machine learning algorithms",
        "deep learning neural networks",
        "python programming",
        "python functions",
    ]

    labels = [
        "ai",
        "ai",
        "programming",
        "programming",
    ]

    extractor = TextFeatureExtractor()
    features = extractor.fit_transform(texts)

    classifier = DocumentClassifier()
    classifier.fit(features, labels)

    model_path = tmp_path / "model.pkl"

    save_model(classifier, model_path)

    loaded_classifier = load_model(model_path)

    prediction = loaded_classifier.predict(
        extractor.transform(["machine learning model"])
    )

    assert prediction[0] == "ai"