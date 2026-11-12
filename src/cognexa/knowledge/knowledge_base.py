from pathlib import Path
from typing import Any

from cognexa.knowledge.ingestion.pipeline import IngestionPipeline
from cognexa.knowledge.retrieval.candidates.dense_retriever import Retriever
from cognexa.knowledge.retrieval.candidates.sparse_retriever import BM25Retriever
from cognexa.knowledge.retrieval.contracts import DocumentChunk


class KnowledgeBase:
    """
    High-level interface for managing Cognexa knowledge.

    Provides document ingestion, dense retrieval,
    sparse retrieval, and parent context expansion.
    """

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        retriever: Retriever,
        sparse_retriever: BM25Retriever | None = None,
    ):
        self.ingestion_pipeline = ingestion_pipeline
        self.retriever = retriever
        self.sparse_retriever = sparse_retriever

    def add_document(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        Add a document and synchronize the sparse index.
        """
        result = self.ingestion_pipeline.ingest(file_path)

        child_count = result.get(
            "child_count",
            result.get("chunk_count", 0),
        )

        if self.sparse_retriever is not None and child_count > 0:
            self.refresh_sparse_index()

        return result

    def refresh_sparse_index(
        self,
        batch_size: int = 100,
    ) -> int:
        """
        Rebuild BM25 index from all child documents in ChromaDB.

        Returns:
            Number of documents loaded into the sparse index.
        """
        if self.sparse_retriever is None:
            raise RuntimeError("Sparse retriever has not been configured.")

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        vector_store = self.retriever.vector_store

        documents: list[DocumentChunk] = []
        offset = 0

        while True:
            records = vector_store.get_all_documents(
                limit=batch_size,
                offset=offset,
            )

            if not records:
                break

            for record in records:
                metadata = record.get("metadata") or {}

                documents.append(
                    DocumentChunk(
                        chunk_id=record["id"],
                        document_id=metadata["document_id"],
                        content=record["content"],
                        metadata=metadata,
                        parent_chunk_id=metadata.get("parent_id"),
                        position=metadata.get("chunk_index"),
                    )
                )

            offset += len(records)

            if len(records) < batch_size:
                break

        self.sparse_retriever.replace_documents(documents)

        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant child chunks using dense retrieval.
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def expand_to_parents(
        self,
        child_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Expand selected child chunks into their parent contexts.

        Parent documents are deduplicated by parent_id.
        Legacy child chunks without parent_id are preserved.
        """

        parent_ids = []
        seen_parent_ids = set()
        fallback_documents = []

        for child in child_documents:
            metadata = child.get("metadata") or {}
            parent_id = metadata.get("parent_id")

            if not parent_id:
                fallback_documents.append(child)
                continue

            if parent_id not in seen_parent_ids:
                parent_ids.append(parent_id)
                seen_parent_ids.add(parent_id)

        parent_documents = (
            self.retriever.vector_store.get_parent_documents(
                ids=parent_ids,
            )
        )

        return parent_documents + fallback_documents

