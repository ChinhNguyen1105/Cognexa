import pytest

from cognexa.knowledge.embedding.embedder import TextEmbedder


class FakeArray:
    """Simulate a NumPy array returned by SentenceTransformer."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


class FakeSentenceTransformer:
    """Simulate SentenceTransformer without loading a real model."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.received_texts = None
        self.encode_kwargs = None

    def encode(self, texts, **kwargs):
        self.received_texts = texts
        self.encode_kwargs = kwargs

        return FakeArray([
            [0.1, 0.2, 0.3]
            for _ in texts
        ])


class TestTextEmbedder:

    def setup_method(self):
        self.fake_model = FakeSentenceTransformer("fake-model")

    def create_embedder(self, monkeypatch, **kwargs):
        fake_model = self.fake_model

        def fake_model_factory(model_name):
            fake_model.model_name = model_name
            return fake_model

        monkeypatch.setattr(
            "cognexa.knowledge.embedding.embedder.SentenceTransformer",
            fake_model_factory,
        )

        return TextEmbedder(**kwargs)

    def test_initializes_model(self, monkeypatch):
        # Arrange & Act
        embedder = self.create_embedder(monkeypatch)

        # Assert
        assert embedder.model is self.fake_model
        assert embedder.model_name == TextEmbedder.DEFAULT_MODEL

    def test_initializes_custom_model(self, monkeypatch):
        # Arrange
        model_name = "custom-embedding-model"

        # Act
        embedder = self.create_embedder(
            monkeypatch,
            model_name=model_name,
        )

        # Assert
        assert embedder.model_name == model_name
        assert self.fake_model.model_name == model_name

    def test_embed_single_text(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)
        text = "Cognexa is an AI project."

        # Act
        vector = embedder.embed_text(text)

        # Assert
        assert vector == [0.1, 0.2, 0.3]
        assert self.fake_model.received_texts == [text]

    def test_embed_multiple_texts(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        texts = [
            "I am learning AI.",
            "I am studying machine learning.",
        ]

        # Act
        vectors = embedder.embed_texts(texts)

        # Assert
        assert len(vectors) == 2
        assert vectors == [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ]

        assert self.fake_model.received_texts == texts

    def test_embed_empty_list(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act
        vectors = embedder.embed_texts([])

        # Assert
        assert vectors == []
        assert self.fake_model.received_texts is None

    def test_embed_empty_text_raises_error(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act & Assert
        with pytest.raises(ValueError, match="cannot be empty"):
            embedder.embed_text("")

    def test_embed_whitespace_text_raises_error(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act & Assert
        with pytest.raises(ValueError, match="cannot be empty"):
            embedder.embed_text("   ")

    def test_embed_non_string_input_raises_error(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act & Assert
        with pytest.raises(TypeError, match="must be a string"):
            embedder.embed_text(123)

    def test_embed_texts_non_list_input_raises_error(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act & Assert
        with pytest.raises(TypeError, match="must be a list"):
            embedder.embed_texts("Cognexa")

    def test_embed_texts_with_empty_item_raises_error(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act & Assert
        with pytest.raises(ValueError, match="index 1"):
            embedder.embed_texts([
                "Valid text",
                "",
            ])

    def test_embed_texts_with_non_string_item_raises_error(
        self,
        monkeypatch,
    ):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act & Assert
        with pytest.raises(TypeError, match="index 1"):
            embedder.embed_texts([
                "Valid text",
                123,
            ])

    def test_passes_normalization_setting_to_model(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(
            monkeypatch,
            normalize_embeddings=False,
        )

        # Act
        embedder.embed_text("Cognexa AI")

        # Assert
        assert (
            self.fake_model.encode_kwargs["normalize_embeddings"]
            is False
        )

    def test_disables_progress_bar(self, monkeypatch):
        # Arrange
        embedder = self.create_embedder(monkeypatch)

        # Act
        embedder.embed_text("Cognexa AI")

        # Assert
        assert (
            self.fake_model.encode_kwargs["show_progress_bar"]
            is False
        )
