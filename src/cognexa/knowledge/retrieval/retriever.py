from typing import Any

from cognexa.knowledge.embedding.embedder import TextEmbedder
from cognexa.knowledge.vector_store.store import VectorStore


class Retriever:
    """
    Retrieve relevant documents using semantic search.

    Combines TextEmbedder and VectorStore to convert
    natural language queries into relevant documents.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents relevant to a natural language query.

        Args:
            query: User's search query.
            top_k: Maximum number of documents to retrieve.

        Returns:
            A list of matching documents with metadata and distance.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results
