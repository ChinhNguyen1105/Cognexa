from __future__ import annotations

from typing import Any

from cognexa.knowledge.embedding.embedder import TextEmbedder
from cognexa.knowledge.vector_store.store import VectorStore

from ..contracts import (
    DocumentChunk,
    RetrievalHit,
    RetrievalQuery,
    RetrievalRun,
    RetrieverType,
)


class Retriever:
    """
    Legacy Dense Retriever.

    Kept temporarily for compatibility with the existing KnowledgeBase
    and CLI implementation.
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
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_embedding = self.embedder.embed_text(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )


class DenseRetriever:
    """
    Advanced Dense Retriever.

    Converts the existing VectorStore results into the shared
    RetrievalRun contract.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    @staticmethod
    def _to_document_chunk(
        result: dict[str, Any],
    ) -> DocumentChunk:
        if not isinstance(result, dict):
            raise TypeError("VectorStore result must be a dictionary")

        metadata = result.get("metadata") or {}

        if not isinstance(metadata, dict):
            raise TypeError("VectorStore metadata must be a dictionary")

        chunk_id = result.get("id") or result.get("chunk_id")
        content = result.get("document") or result.get("content")

        if not chunk_id:
            raise ValueError("VectorStore result is missing chunk ID")

        if not content:
            raise ValueError(
                f"VectorStore result {chunk_id!r} is missing content"
            )

        document_id = metadata.get("document_id") or chunk_id

        parent_chunk_id = (
            metadata.get("parent_id")
            or metadata.get("parent_chunk_id")
        )

        position = metadata.get(
            "chunk_index",
            metadata.get("position"),
        )

        if isinstance(position, bool) or not isinstance(position, int):
            position = None

        return DocumentChunk(
            chunk_id=str(chunk_id),
            document_id=str(document_id),
            content=content,
            metadata=metadata,
            parent_chunk_id=(
                str(parent_chunk_id)
                if parent_chunk_id is not None
                else None
            ),
            position=position,
        )

    @staticmethod
    def _get_score(result: dict[str, Any]) -> tuple[float, str]:
        if "distance" in result:
            return float(result["distance"]), "distance"

        if "score" in result:
            return float(result["score"]), "similarity"

        raise ValueError(
            "VectorStore result must contain either 'distance' or 'score'"
        )

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
    ) -> RetrievalRun:
        if not isinstance(query, RetrievalQuery):
            raise TypeError("query must be a RetrievalQuery")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_embedding = self.embedder.embed_text(query.text)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        hits: list[RetrievalHit] = []

        for rank, result in enumerate(results, start=1):
            chunk = self._to_document_chunk(result)
            raw_score, score_type = self._get_score(result)

            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    query_id=query.query_id,
                    retriever=RetrieverType.DENSE,
                    rank=rank,
                    raw_score=raw_score,
                    score_type=score_type,
                )
            )

        return RetrievalRun(
            query_id=query.query_id,
            retriever=RetrieverType.DENSE,
            hits=hits,
        )