from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    """
    Persistent vector storage powered by ChromaDB.

    Stores searchable child chunks and their embeddings
    separately from parent context documents.
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/vector_store",
        collection_name: str = "cognexa_knowledge",
    ):
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty.")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.parent_collection_name = f"{collection_name}_parents"

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.parent_collection = self.client.get_or_create_collection(
            name=self.parent_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _validate_documents(
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None,
    ) -> None:
        if not ids:
            raise ValueError("ids cannot be empty.")

        if not (
            len(ids) == len(documents) == len(embeddings)
        ):
            raise ValueError(
                "ids, documents, and embeddings must have equal lengths."
            )

        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")

        if any(
            not isinstance(doc_id, str) or not doc_id.strip()
            for doc_id in ids
        ):
            raise ValueError(
                "Every document ID must be a non-empty string."
            )

        if any(
            not isinstance(document, str) or not document.strip()
            for document in documents
        ):
            raise ValueError(
                "Every document must be a non-empty string."
            )

        if any(not embedding for embedding in embeddings):
            raise ValueError("Embeddings cannot be empty.")

        if any(
            not isinstance(value, (int, float))
            for embedding in embeddings
            for value in embedding
        ):
            raise ValueError("Embedding values must be numeric.")

        dimensions = {
            len(embedding)
            for embedding in embeddings
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All embeddings must have the same dimensions."
            )

        if metadatas is not None:
            if len(metadatas) != len(ids):
                raise ValueError(
                    "metadatas must have the same length as ids."
                )

            if any(
                not isinstance(metadata, dict)
                for metadata in metadatas
            ):
                raise TypeError(
                    "Each metadata item must be a dictionary."
                )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add searchable child documents."""

        self._validate_documents(
            ids,
            documents,
            embeddings,
            metadatas,
        )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def add_parent_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store parent context documents."""

        self._validate_documents(
            ids,
            documents,
            embeddings,
            metadatas,
        )

        self.parent_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def get_parent_documents(
        self,
        ids: list[str],
    ) -> list[dict[str, Any]]:
        """Retrieve parent documents by their IDs."""

        if not ids:
            return []

        if any(
            not isinstance(doc_id, str) or not doc_id.strip()
            for doc_id in ids
        ):
            raise ValueError(
                "Every parent ID must be a non-empty string."
            )

        if len(set(ids)) != len(ids):
            raise ValueError("Parent IDs must be unique.")

        results = self.parent_collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        documents = results["documents"] or []
        metadatas = results["metadatas"] or []
        result_ids = results["ids"]

        parent_map = {
            doc_id: {
                "id": doc_id,
                "content": document,
                "metadata": metadata,
            }
            for doc_id, document, metadata in zip(
                result_ids,
                documents,
                metadatas,
            )
        }

        # Preserve the order of requested IDs.
        return [
            parent_map[parent_id]
            for parent_id in ids
            if parent_id in parent_map
        ]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for child documents using vector similarity."""

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "id": doc_id,
                "content": document,
                "metadata": metadata,
                "distance": distance,
            }
            for doc_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
            )
        ]

    def delete(self, ids: list[str]) -> None:
        """Delete child documents by their identifiers."""

        if not ids:
            raise ValueError("ids cannot be empty.")

        if any(
            not isinstance(doc_id, str) or not doc_id.strip()
            for doc_id in ids
        ):
            raise ValueError(
                "Every document ID must be a non-empty string."
            )

        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")

        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Return the number of child documents."""

        return self.collection.count()

    def parent_count(self) -> int:
        """Return the number of parent documents."""

        return self.parent_collection.count()

    def reset(self) -> None:
        """Delete and recreate both collections."""

        self.client.delete_collection(
            name=self.collection_name
        )

        self.client.delete_collection(
            name=self.parent_collection_name
        )

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.parent_collection = self.client.create_collection(
            name=self.parent_collection_name,
            metadata={"hnsw:space": "cosine"},
        )