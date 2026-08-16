from cognexa.ml.classical.classifier import DocumentClassifier
from cognexa.ml.classical.features import TextFeatureExtractor


def test_document_classifier():
    texts = [
        "machine learning algorithms",
        "deep neural networks",
        "python programming language",
        "python functions and classes",
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

    prediction = classifier.predict(
        extractor.transform(["machine learning model"])
    )

    assert prediction[0] == "ai"