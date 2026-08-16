from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    """Convert text documents into numerical TF-IDF features."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts: list[str]):
        """Learn vocabulary and transform texts into feature vectors."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        """Transform new texts using the learned vocabulary."""
        return self.vectorizer.transform(texts)