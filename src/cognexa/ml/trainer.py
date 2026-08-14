from cognexa.ml.classifier import DocumentClassifier
from cognexa.ml.features import TextFeatureExtractor


class DocumentTrainer:
    """Train a document classification model."""

    def __init__(self):
        self.feature_extractor = TextFeatureExtractor()
        self.classifier = DocumentClassifier()

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Train the feature extractor and classifier."""
        features = self.feature_extractor.fit_transform(texts)
        self.classifier.fit(features, labels)

    def predict(self, texts: list[str]) -> list[str]:
        """Predict labels for new documents."""
        features = self.feature_extractor.transform(texts)
        return self.classifier.predict(features)