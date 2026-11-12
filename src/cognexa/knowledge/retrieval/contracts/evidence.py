from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ._validation import (
    require_finite_number,
    require_text,
    require_unit_interval,
)
from .document import DocumentChunk


class EvidenceType(str, Enum):
    PRIMARY = "primary"
    PARENT = "parent"
    NEIGHBOR = "neighbor"


@dataclass
class Evidence:
    evidence_id: str
    chunk: DocumentChunk

    evidence_type: EvidenceType
    source_chunk_id: str

    related_query_ids: list[str] = field(default_factory=list)
    relevance_score: float | None = None

    def __post_init__(self) -> None:
        require_text(self.evidence_id, "evidence_id")
        require_text(self.source_chunk_id, "source_chunk_id")

        for query_id in self.related_query_ids:
            require_text(query_id, "related_query_id")

        if self.relevance_score is not None:
            require_finite_number(
                self.relevance_score,
                "relevance_score",
            )


@dataclass
class SubQuestionCoverage:
    query_id: str
    question: str

    coverage_score: float
    covered: bool

    evidence_ids: list[str] = field(default_factory=list)
    missing_information: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        require_text(self.query_id, "query_id")
        require_text(self.question, "question")
        require_unit_interval(self.coverage_score, "coverage_score")

        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")

        if self.covered and not self.evidence_ids:
            raise ValueError("Covered questions must reference evidence")

        if self.missing_information is not None:
            require_text(self.missing_information, "missing_information")


@dataclass
class CoverageReport:
    overall_score: float
    sub_question_results: list[SubQuestionCoverage] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        require_unit_interval(self.overall_score, "overall_score")

        query_ids = [
            result.query_id
            for result in self.sub_question_results
        ]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Sub-question query IDs must be unique")


@dataclass(frozen=True)
class SufficiencyDecision:
    sufficient: bool
    missing_query_ids: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        for query_id in self.missing_query_ids:
            require_text(query_id, "missing_query_id")

        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")

        if self.sufficient and self.missing_query_ids:
            raise ValueError(
                "A sufficient decision cannot contain missing query IDs"
            )