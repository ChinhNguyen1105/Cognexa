from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Convert text into dense numerical vectors.

    Uses a multilingual Sentence Transformer model
    to support semantic search across multiple languages.
    """

    DEFAULT_MODEL = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single text into an embedding vector.

        Args:
            text: Input text.

        Returns:
            A list of floating-point numbers.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            raise ValueError("text cannot be empty.")

        embeddings = self.embed_texts([text])

        return embeddings[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple texts into embedding vectors.

        Args:
            texts: A list of input texts.

        Returns:
            A list of embedding vectors.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")

        if not texts:
            return []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Item at index {index} must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"Text at index {index} cannot be empty."
                )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embeddings.tolist()
