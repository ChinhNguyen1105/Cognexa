from pathlib import Path
from typing import Any

from cognexa.knowledge.ingestion.pipeline import IngestionPipeline
from cognexa.knowledge.retrieval.retriever import Retriever


class KnowledgeBase:
    """
    High-level interface for managing Cognexa knowledge.

    Provides document ingestion, semantic retrieval,
    and parent context expansion.
    """

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        retriever: Retriever,
    ):
        self.ingestion_pipeline = ingestion_pipeline
        self.retriever = retriever

    def add_document(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        Add a document to the knowledge base.
        """
        return self.ingestion_pipeline.ingest(file_path)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant child chunks.
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
