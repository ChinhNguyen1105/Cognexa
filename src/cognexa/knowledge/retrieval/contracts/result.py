from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ._validation import require_positive_int, require_text
from .evidence import (
    CoverageReport,
    Evidence,
    SufficiencyDecision,
)
from .query import QueryPlan


class RetrievalStopReason(str, Enum):
    THRESHOLD_REACHED = "threshold_reached"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    NO_NEW_EVIDENCE = "no_new_evidence"
    RETRIEVAL_BUDGET_EXHAUSTED = "retrieval_budget_exhausted"


@dataclass
class RetrievalResult:
    query_plan: QueryPlan
    evidence: list[Evidence]

    coverage: CoverageReport
    decision: SufficiencyDecision

    iterations: int
    stop_reason: RetrievalStopReason

    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_positive_int(self.iterations, "iterations")

        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")

        evidence_ids = [
            item.evidence_id
            for item in self.evidence
        ]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique")

        if (
            self.stop_reason == RetrievalStopReason.THRESHOLD_REACHED
            and not self.decision.sufficient
        ):
            raise ValueError(
                "threshold_reached requires a sufficient decision"
            )

    @property
    def sufficient(self) -> bool:
        return self.decision.sufficient

    @property
    def original_query(self) -> str:
        return self.query_plan.original_query