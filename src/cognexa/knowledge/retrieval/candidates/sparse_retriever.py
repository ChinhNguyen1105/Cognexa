from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

from ..contracts import (
    DocumentChunk,
    RetrievalHit,
    RetrievalQuery,
    RetrievalRun,
    RetrieverType,
)


def tokenize(text: str) -> list[str]:
    """
    Basic case-insensitive tokenizer.

    Preserves Unicode word characters and technical identifiers.
    """
    return re.findall(r"[\w]+", text.casefold())


class BM25Retriever:
    """
    In-memory BM25 candidate retriever.

    The index is built from DocumentChunk objects.
    It can be replaced or updated when the Knowledge Base changes.
    """

    def __init__(
        self,
        documents: Sequence[DocumentChunk] = (),
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")

        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")

        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

        self._documents: dict[str, DocumentChunk] = {}
        self._indexed_documents: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._index: BM25Okapi | None = None

        self.replace_documents(documents)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def indexed_document_count(self) -> int:
        return len(self._indexed_documents)

    def replace_documents(
        self,
        documents: Sequence[DocumentChunk],
    ) -> None:
        new_documents: dict[str, DocumentChunk] = {}

        for document in documents:
            if not isinstance(document, DocumentChunk):
                raise TypeError(
                    "All indexed documents must be DocumentChunk instances"
                )

            if document.chunk_id in new_documents:
                raise ValueError(
                    f"Duplicate chunk ID: {document.chunk_id}"
                )

            new_documents[document.chunk_id] = document

        self._documents = new_documents
        self._rebuild_index()

    def upsert_documents(
        self,
        documents: Sequence[DocumentChunk],
    ) -> None:
        incoming_ids: set[str] = set()

        for document in documents:
            if not isinstance(document, DocumentChunk):
                raise TypeError(
                    "All indexed documents must be DocumentChunk instances"
                )

            if document.chunk_id in incoming_ids:
                raise ValueError(
                    f"Duplicate chunk ID in update: {document.chunk_id}"
                )

            incoming_ids.add(document.chunk_id)

        for document in documents:
            self._documents[document.chunk_id] = document

        self._rebuild_index()

    def remove_documents(
        self,
        chunk_ids: Sequence[str],
    ) -> int:
        removed = 0

        for chunk_id in chunk_ids:
            if self._documents.pop(chunk_id, None) is not None:
                removed += 1

        if removed:
            self._rebuild_index()

        return removed

    def _rebuild_index(self) -> None:
        self._indexed_documents = []
        self._tokenized_corpus = []
        self._index = None

        for document in self._documents.values():
            tokens = self.tokenizer(document.content)

            if not tokens:
                continue

            self._indexed_documents.append(document)
            self._tokenized_corpus.append(tokens)

        if not self._tokenized_corpus:
            return

        self._index = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
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

        if self._index is None:
            return RetrievalRun(
                query_id=query.query_id,
                retriever=RetrieverType.BM25,
                hits=[],
            )

        query_tokens = self.tokenizer(query.text)

        if not query_tokens:
            return RetrievalRun(
                query_id=query.query_id,
                retriever=RetrieverType.BM25,
                hits=[],
            )

        scores = self._index.get_scores(query_tokens)

        ranked_results = [
            (index, float(score))
            for index, score in enumerate(scores)
            if score > 0
        ]

        ranked_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        hits: list[RetrievalHit] = []

        for rank, (index, score) in enumerate(
            ranked_results[:top_k],
            start=1,
        ):
            hits.append(
                RetrievalHit(
                    chunk=self._indexed_documents[index],
                    query_id=query.query_id,
                    retriever=RetrieverType.BM25,
                    rank=rank,
                    raw_score=score,
                    score_type="bm25",
                )
            )

        return RetrievalRun(
            query_id=query.query_id,
            retriever=RetrieverType.BM25,
            hits=hits,
        )