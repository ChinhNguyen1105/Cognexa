from __future__ import annotations

from typing import Protocol, Sequence

from .contracts.candidate import (
    FusedCandidate,
    RankedCandidate,
    RetrievalRun,
)
from .contracts.evidence import (
    CoverageReport,
    Evidence,
    SufficiencyDecision,
)
from .contracts.query import QueryPlan, RetrievalQuery
from .contracts.result import RetrievalResult


class QueryUnderstandingProtocol(Protocol):
    def analyze(self, query: str) -> QueryPlan:
        ...


class CandidateRetrieverProtocol(Protocol):
    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int,
    ) -> RetrievalRun:
        ...


class ResultFusionProtocol(Protocol):
    def fuse(
        self,
        runs: Sequence[RetrievalRun],
    ) -> list[FusedCandidate]:
        ...


class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[FusedCandidate],
        top_k: int | None = None,
    ) -> list[RankedCandidate]:
        ...


class EvidenceExpanderProtocol(Protocol):
    def expand(
        self,
        candidates: Sequence[RankedCandidate],
    ) -> list[Evidence]:
        ...


class CoverageAnalyzerProtocol(Protocol):
    def evaluate(
        self,
        query_plan: QueryPlan,
        evidence: Sequence[Evidence],
    ) -> CoverageReport:
        ...


class SufficiencyCheckerProtocol(Protocol):
    def check(
        self,
        report: CoverageReport,
    ) -> SufficiencyDecision:
        ...


class AdaptiveRetrievalProtocol(Protocol):
    def retrieve(
        self,
        query_plan: QueryPlan,
    ) -> RetrievalResult:
        ...