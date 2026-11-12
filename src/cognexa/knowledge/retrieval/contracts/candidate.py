from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ._validation import (
    require_finite_number,
    require_positive_int,
    require_text,
)
from .document import DocumentChunk


class RetrieverType(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"


@dataclass(frozen=True)
class RetrievalHit:
    chunk: DocumentChunk
    query_id: str
    retriever: RetrieverType

    rank: int
    raw_score: float
    score_type: str

    def __post_init__(self) -> None:
        require_text(self.query_id, "query_id")
        require_text(self.score_type, "score_type")
        require_positive_int(self.rank, "rank")
        require_finite_number(self.raw_score, "raw_score")


@dataclass
class RetrievalRun:
    query_id: str
    retriever: RetrieverType
    hits: list[RetrievalHit] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_text(self.query_id, "query_id")

        for hit in self.hits:
            if hit.query_id != self.query_id:
                raise ValueError("All hits must belong to the same query_id")

            if hit.retriever != self.retriever:
                raise ValueError("All hits must use the same retriever")

        chunk_ids = [hit.chunk.chunk_id for hit in self.hits]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("A RetrievalRun cannot contain duplicate chunks")


@dataclass(frozen=True)
class RetrievalMatch:
    query_id: str
    retriever: RetrieverType
    rank: int
    raw_score: float
    score_type: str

    def __post_init__(self) -> None:
        require_text(self.query_id, "query_id")
        require_text(self.score_type, "score_type")
        require_positive_int(self.rank, "rank")
        require_finite_number(self.raw_score, "raw_score")


@dataclass
class FusedCandidate:
    chunk: DocumentChunk
    rrf_score: float
    matches: list[RetrievalMatch]
    fusion_rank: int

    def __post_init__(self) -> None:
        require_finite_number(self.rrf_score, "rrf_score")
        require_positive_int(self.fusion_rank, "fusion_rank")

        if self.rrf_score <= 0:
            raise ValueError("rrf_score must be greater than zero")

        if not self.matches:
            raise ValueError("matches cannot be empty")


@dataclass(frozen=True)
class RankedCandidate:
    query_id: str
    candidate: FusedCandidate
    rerank_score: float
    rerank_rank: int

    def __post_init__(self) -> None:
        require_text(self.query_id, "query_id")
        require_finite_number(self.rerank_score, "rerank_score")
        require_positive_int(self.rerank_rank, "rerank_rank")