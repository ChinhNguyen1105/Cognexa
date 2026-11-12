from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ._validation import require_finite_number, require_text


class QueryType(str, Enum):
    ORIGINAL = "original"
    REWRITTEN = "rewritten"
    SUB_QUESTION = "sub_question"


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    text: str
    query_type: QueryType = QueryType.ORIGINAL
    weight: float = 1.0
    aspect: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        require_text(self.query_id, "query_id")
        require_text(self.text, "text")
        require_finite_number(self.weight, "weight")

        if self.weight <= 0:
            raise ValueError("weight must be greater than zero")

        if self.aspect is not None:
            require_text(self.aspect, "aspect")


@dataclass
class QueryPlan:
    original_query: str
    normalized_query: str
    intent: str
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_text(self.original_query, "original_query")
        require_text(self.normalized_query, "normalized_query")
        require_text(self.intent, "intent")

        if not self.retrieval_queries:
            raise ValueError("retrieval_queries cannot be empty")

        query_ids = [query.query_id for query in self.retrieval_queries]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError("retrieval query IDs must be unique")

        if not any(
            query.query_type == QueryType.ORIGINAL
            for query in self.retrieval_queries
        ):
            raise ValueError("QueryPlan must contain the original query")

        for question in self.sub_questions:
            require_text(question, "sub_question")