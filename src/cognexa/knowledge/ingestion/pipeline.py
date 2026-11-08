from pathlib import Path
from typing import Any
from uuid import uuid4

from cognexa.knowledge.ingestion.loader import DocumentLoader
from cognexa.knowledge.processing.chunker import TextChunker
from cognexa.knowledge.embedding.embedder import TextEmbedder
from cognexa.knowledge.vector_store.store import VectorStore


class IngestionPipeline:
    """
    Coordinate the parent-child document ingestion process.

    Pipeline:
        DocumentLoader
        -> Parent Chunking
        -> Child Chunking
        -> TextEmbedder
        -> VectorStore
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 150,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

        self.parent_chunker = TextChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )

    def ingest(
        self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        Load, split, embed, and store a document.

        Child chunks are searchable.
        Parent chunks are used for context expansion.
        """

        document = self.loader.load(file_path)

        content = document["content"]
        source_metadata = document["metadata"]

        hierarchy = self.parent_chunker.chunk_document(
            text=content,
            child_chunker=self.chunker,
        )

        if not hierarchy:
            return {
                "source": source_metadata["source"],
                "file_type": source_metadata["file_type"],
                "chunk_count": 0,
                "parent_count": 0,
                "child_count": 0,
                "parent_ids": [],
                "child_ids": [],
            }

        document_id = str(uuid4())

        parent_ids = []
        parent_documents = []
        parent_metadatas = []

        child_ids = []
        child_documents = []
        child_metadatas = []

        for parent in hierarchy:
            parent_index = parent["parent_index"]

            parent_id = (
                f"{document_id}_parent_{parent_index}"
            )

            parent_ids.append(parent_id)
            parent_documents.append(parent["content"])

            parent_metadatas.append(
                {
                    "document_id": document_id,
                    "source": source_metadata["source"],
                    "file_name": source_metadata["file_name"],
                    "file_type": source_metadata["file_type"],
                    "parent_index": parent_index,
                }
            )

            for child in parent["children"]:
                child_id = str(uuid4())

                child_ids.append(child_id)
                child_documents.append(child["content"])

                child_metadatas.append(
                    {
                        "document_id": document_id,
                        "source": source_metadata["source"],
                        "file_name": source_metadata["file_name"],
                        "file_type": source_metadata["file_type"],
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "chunk_index": child["chunk_index"],
                    }
                )

        # Generate embeddings separately for child and parent.
        child_embeddings = self.embedder.embed_texts(
            child_documents
        )

        parent_embeddings = self.embedder.embed_texts(
            parent_documents
        )

        # Store searchable child chunks.
        self.vector_store.add_documents(
            ids=child_ids,
            documents=child_documents,
            embeddings=child_embeddings,
            metadatas=child_metadatas,
        )

        # Store parent context documents.
        self.vector_store.add_parent_documents(
            ids=parent_ids,
            documents=parent_documents,
            embeddings=parent_embeddings,
            metadatas=parent_metadatas,
        )

        return {
            "source": source_metadata["source"],
            "file_type": source_metadata["file_type"],
            "chunk_count": len(child_ids),
            "parent_count": len(parent_ids),
            "child_count": len(child_ids),
            "parent_ids": parent_ids,
            "child_ids": child_ids,
        }