from sklearn.linear_model import LogisticRegression


class DocumentClassifier:
    """Classify documents using a machine learning model."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, features, labels: list[str]) -> None:
        """Train the classifier."""
        self.model.fit(features, labels)

    def predict(self, features) -> list[str]:
        """Predict labels."""
        return self.model.predict(features).tolist()