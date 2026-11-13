from __future__ import annotations

from collections.abc import Sequence

from cognexa.knowledge.retrieval.contracts.evidence import (
    Evidence,
    EvidenceType,
)
from cognexa.knowledge.retrieval.contracts.query import QueryPlan
from cognexa.knowledge.retrieval.contracts.result import (
    RetrievalResult,
    RetrievalStopReason,
)
from cognexa.knowledge.retrieval.interfaces import (
    CandidateRetrieverProtocol,
    ResultFusionProtocol,
    RerankerProtocol,
    EvidenceExpanderProtocol,
    CoverageAnalyzerProtocol,
    SufficiencyCheckerProtocol,
)


class AdaptiveRetriever:
    """
    Orchestrate the complete retrieval pipeline.

    Pipeline:
        Dense + BM25
            -> RRF Fusion
            -> Cross-Encoder Reranking
            -> Evidence Expansion
            -> Evidence Aggregation
            -> Coverage Analysis
            -> Sufficiency Check

    Retrieval is repeated with an increasing top_k until:
        1. The coverage threshold is reached.
        2. No new evidence is discovered.
        3. Maximum iterations are reached.
    """

    _TYPE_PRIORITY = {
        EvidenceType.NEIGHBOR: 1,
        EvidenceType.PARENT: 2,
        EvidenceType.PRIMARY: 3,
    }

    def __init__(
        self,
        dense_retriever: CandidateRetrieverProtocol,
        sparse_retriever: CandidateRetrieverProtocol,
        fusion: ResultFusionProtocol,
        reranker: RerankerProtocol,
        expander: EvidenceExpanderProtocol,
        coverage_analyzer: CoverageAnalyzerProtocol,
        sufficiency_checker: SufficiencyCheckerProtocol,
        top_k: int = 5,
        max_iterations: int = 3,
        candidate_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> None:
        self._validate_positive_int(top_k, "top_k")
        self._validate_positive_int(max_iterations, "max_iterations")
        if candidate_top_k is not None:
            self._validate_positive_int(candidate_top_k, "candidate_top_k")
        if rerank_top_k is not None:
            self._validate_positive_int(rerank_top_k, "rerank_top_k")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.expander = expander
        self.coverage_analyzer = coverage_analyzer
        self.sufficiency_checker = sufficiency_checker

        self.top_k = top_k
        self.max_iterations = max_iterations
        self.candidate_top_k = candidate_top_k or max(15, top_k)
        self.rerank_top_k = rerank_top_k or min(5, self.candidate_top_k)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")

    def retrieve(self, query_plan: QueryPlan) -> RetrievalResult:
        """Execute adaptive retrieval for a query plan."""

        if not isinstance(query_plan, QueryPlan):
            raise TypeError("query_plan must be a QueryPlan")

        evidence_by_chunk_id: dict[str, Evidence] = {}
        iteration_history: list[dict] = []

        previous_evidence_ids: set[str] = set()

        final_coverage = None
        final_decision = None
        stop_reason = None

        for iteration in range(1, self.max_iterations + 1):
            current_top_k = self.top_k * iteration

            iteration_evidence = self._retrieve_iteration(
                query_plan=query_plan,
                top_k=current_top_k,
            )

            for evidence in iteration_evidence:
                self._merge_evidence(
                    evidence_by_chunk_id,
                    evidence,
                )

            current_evidence_ids = set(evidence_by_chunk_id)

            new_evidence_ids = (
                current_evidence_ids - previous_evidence_ids
            )

            previous_evidence_ids = current_evidence_ids

            all_evidence = list(evidence_by_chunk_id.values())

            final_coverage = self.coverage_analyzer.evaluate(
                query_plan=query_plan,
                evidence=all_evidence,
            )

            final_decision = self.sufficiency_checker.check(
                report=final_coverage,
            )

            iteration_history.append(
                {
                    "iteration": iteration,
                    "top_k": current_top_k,
                    "new_evidence_count": len(new_evidence_ids),
                    "total_evidence_count": len(all_evidence),
                    "coverage_score": final_coverage.overall_score,
                    "sufficient": final_decision.sufficient,
                }
            )

            # Reaching the threshold takes priority over other
            # stopping conditions.
            if final_decision.sufficient:
                stop_reason = RetrievalStopReason.THRESHOLD_REACHED
                break

            if not new_evidence_ids:
                stop_reason = RetrievalStopReason.NO_NEW_EVIDENCE
                break

            if iteration == self.max_iterations:
                stop_reason = (
                    RetrievalStopReason.MAX_ITERATIONS_REACHED
                )
                break

        return RetrievalResult(
            query_plan=query_plan,
            evidence=list(evidence_by_chunk_id.values()),
            coverage=final_coverage,
            decision=final_decision,
            iterations=len(iteration_history),
            stop_reason=stop_reason,
            diagnostics={
                "base_top_k": self.top_k,
                "max_iterations": self.max_iterations,
                "iteration_history": iteration_history,
            },
        )

    def _retrieve_iteration(
        self,
        query_plan: QueryPlan,
        top_k: int,
    ) -> list[Evidence]:
        """Execute the retrieval pipeline for every retrieval query."""

        all_evidence: list[Evidence] = []

        for query in query_plan.retrieval_queries:
            dense_run = self.dense_retriever.retrieve(
                query=query,
                top_k=max(self.candidate_top_k, top_k),
            )

            sparse_run = self.sparse_retriever.retrieve(
                query=query,
                top_k=max(self.candidate_top_k, top_k),
            )

            fused_candidates = self.fusion.fuse(
                runs=[dense_run, sparse_run],
            )

            ranked_candidates = self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                top_k=min(self.rerank_top_k, len(fused_candidates) or self.rerank_top_k),
            )

            expanded_evidence = self.expander.expand(
                candidates=ranked_candidates,
            )

            all_evidence.extend(expanded_evidence)

        return all_evidence

    def _merge_evidence(
        self,
        evidence_by_chunk_id: dict[str, Evidence],
        incoming: Evidence,
    ) -> None:
        """
        Merge evidence from different queries and iterations.

        Evidence is deduplicated by chunk_id.
        Related query IDs are combined to preserve coverage information.
        The highest-priority evidence type is retained.
        """

        chunk_id = incoming.chunk.chunk_id
        existing = evidence_by_chunk_id.get(chunk_id)

        if existing is None:
            evidence_by_chunk_id[chunk_id] = incoming
            return

        related_query_ids = list(existing.related_query_ids)

        for query_id in incoming.related_query_ids:
            if query_id not in related_query_ids:
                related_query_ids.append(query_id)

        existing_priority = self._TYPE_PRIORITY[
            existing.evidence_type
        ]
        incoming_priority = self._TYPE_PRIORITY[
            incoming.evidence_type
        ]

        selected = (
            incoming
            if incoming_priority > existing_priority
            else existing
        )

        relevance_score = existing.relevance_score

        if (
            incoming.relevance_score is not None
            and (
                relevance_score is None
                or incoming.relevance_score > relevance_score
            )
        ):
            relevance_score = incoming.relevance_score

        evidence_by_chunk_id[chunk_id] = Evidence(
            evidence_id=selected.evidence_id,
            chunk=selected.chunk,
            evidence_type=selected.evidence_type,
            source_chunk_id=selected.source_chunk_id,
            related_query_ids=related_query_ids,
            relevance_score=relevance_score,
        )
