from __future__ import annotations

import pytest

from cognexa.knowledge.retrieval.adaptive_retriever import AdaptiveRetriever
from cognexa.knowledge.retrieval.contracts.candidate import (
    FusedCandidate,
    RankedCandidate,
    RetrievalMatch,
    RetrievalRun,
    RetrieverType,
)
from cognexa.knowledge.retrieval.contracts.document import DocumentChunk
from cognexa.knowledge.retrieval.contracts.evidence import (
    Evidence,
    EvidenceType,
    SufficiencyDecision,
)
from cognexa.knowledge.retrieval.contracts.query import (
    QueryPlan,
    RetrievalQuery,
)
from cognexa.knowledge.retrieval.contracts.result import (
    RetrievalStopReason,
)
from cognexa.knowledge.retrieval.coverage.coverage_analyzer import (
    CoverageAnalyzer,
)
from cognexa.knowledge.retrieval.coverage.sufficiency_checker import (
    SufficiencyChecker,
)


def make_query_plan(query_count: int = 1) -> QueryPlan:
    queries = [
        RetrievalQuery(
            query_id=f"q{i}",
            text=f"question {i}",
        )
        for i in range(1, query_count + 1)
    ]

    return QueryPlan(
        original_query="original question",
        normalized_query="normalized question",
        intent="information",
        retrieval_queries=queries,
    )


class FakeRetriever:
    def __init__(self, retriever_type: RetrieverType) -> None:
        self.retriever_type = retriever_type
        self.calls: list[tuple[str, int]] = []

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int,
    ) -> RetrievalRun:
        self.calls.append((query.query_id, top_k))

        return RetrievalRun(
            query_id=query.query_id,
            retriever=self.retriever_type,
            hits=[],
        )


class FakeFusion:
    def __init__(self, unique_per_call: bool = False) -> None:
        self.unique_per_call = unique_per_call
        self.calls = 0

    def fuse(self, runs) -> list[FusedCandidate]:
        self.calls += 1

        query_id = runs[0].query_id

        chunk_id = (
            f"chunk-{self.calls}"
            if self.unique_per_call
            else "shared-chunk"
        )

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            content="Evidence content",
        )

        return [
            FusedCandidate(
                chunk=chunk,
                rrf_score=0.03,
                matches=[
                    RetrievalMatch(
                        query_id=query_id,
                        retriever=RetrieverType.DENSE,
                        rank=1,
                        raw_score=0.1,
                        score_type="distance",
                    )
                ],
                fusion_rank=1,
            )
        ]


class FakeReranker:
    def rerank(
        self,
        query: RetrievalQuery,
        candidates,
        top_k=None,
    ) -> list[RankedCandidate]:
        selected = list(candidates)

        if top_k is not None:
            selected = selected[:top_k]

        return [
            RankedCandidate(
                query_id=query.query_id,
                candidate=candidate,
                rerank_score=0.9,
                rerank_rank=rank,
            )
            for rank, candidate in enumerate(selected, start=1)
        ]


class FakeExpander:
    def __init__(
        self,
        excluded_query_ids: set[str] | None = None,
    ) -> None:
        self.excluded_query_ids = excluded_query_ids or set()

    def expand(self, candidates) -> list[Evidence]:
        evidence = []

        for ranked in candidates:
            if ranked.query_id in self.excluded_query_ids:
                continue

            chunk = ranked.candidate.chunk

            evidence.append(
                Evidence(
                    evidence_id=chunk.chunk_id,
                    chunk=chunk,
                    evidence_type=EvidenceType.PRIMARY,
                    source_chunk_id=chunk.chunk_id,
                    related_query_ids=[ranked.query_id],
                    relevance_score=ranked.rerank_score,
                )
            )

        return evidence


class AlwaysInsufficient:
    def check(self, report) -> SufficiencyDecision:
        return SufficiencyDecision(
            sufficient=False,
            missing_query_ids=("q1",),
            reason="Test forces insufficient result",
        )


def build_retriever(
    *,
    query_count: int = 1,
    expander=None,
    sufficiency_checker=None,
    unique_per_call: bool = False,
    top_k: int = 5,
    max_iterations: int = 3,
):
    dense = FakeRetriever(RetrieverType.DENSE)
    sparse = FakeRetriever(RetrieverType.BM25)

    retriever = AdaptiveRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=FakeFusion(unique_per_call=unique_per_call),
        reranker=FakeReranker(),
        expander=expander or FakeExpander(),
        coverage_analyzer=CoverageAnalyzer(),
        sufficiency_checker=(
            sufficiency_checker or SufficiencyChecker(min_coverage=1.0)
        ),
        top_k=top_k,
        max_iterations=max_iterations,
    )

    return retriever, dense, sparse, make_query_plan(query_count)


def test_stops_when_coverage_threshold_is_reached():
    retriever, _, _, query_plan = build_retriever()

    result = retriever.retrieve(query_plan)

    assert result.sufficient is True
    assert result.iterations == 1
    assert result.stop_reason == RetrievalStopReason.THRESHOLD_REACHED
    assert result.coverage.overall_score == 1.0


def test_merges_related_query_ids_for_duplicate_evidence():
    retriever, _, _, query_plan = build_retriever(query_count=2)

    result = retriever.retrieve(query_plan)

    assert result.sufficient is True
    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert set(evidence.related_query_ids) == {"q1", "q2"}
    assert evidence.evidence_id == "shared-chunk"


def test_stops_when_no_new_evidence_is_found():
    expander = FakeExpander(excluded_query_ids={"q2"})

    retriever, dense, sparse, query_plan = build_retriever(
        query_count=2,
        expander=expander,
    )

    result = retriever.retrieve(query_plan)

    assert result.sufficient is False
    assert result.iterations == 2
    assert result.stop_reason == RetrievalStopReason.NO_NEW_EVIDENCE
    assert len(result.evidence) == 1

    assert [top_k for _, top_k in dense.calls] == [5, 5, 10, 10]
    assert [top_k for _, top_k in sparse.calls] == [5, 5, 10, 10]


def test_stops_when_max_iterations_is_reached():
    retriever, _, _, query_plan = build_retriever(
        sufficiency_checker=AlwaysInsufficient(),
        unique_per_call=True,
        max_iterations=3,
    )

    result = retriever.retrieve(query_plan)

    assert result.sufficient is False
    assert result.iterations == 3
    assert result.stop_reason == RetrievalStopReason.MAX_ITERATIONS_REACHED
    assert len(result.evidence) == 3

    assert [
        item["top_k"]
        for item in result.diagnostics["iteration_history"]
    ] == [5, 10, 15]


def test_retrieval_result_contains_iteration_diagnostics():
    retriever, _, _, query_plan = build_retriever()

    result = retriever.retrieve(query_plan)

    assert result.diagnostics["base_top_k"] == 5
    assert result.diagnostics["max_iterations"] == 3

    history = result.diagnostics["iteration_history"]

    assert len(history) == 1
    assert history[0]["iteration"] == 1
    assert history[0]["top_k"] == 5
    assert history[0]["new_evidence_count"] == 1
    assert history[0]["sufficient"] is True



@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"top_k": True}, TypeError),
        ({"max_iterations": 2.5}, TypeError),
        ({"top_k": 0}, ValueError),
        ({"max_iterations": -1}, ValueError),
    ],
)
def test_constructor_validates_positive_integer_parameters(kwargs, error):
    with pytest.raises(error):
        build_retriever(**kwargs)


def test_retrieve_rejects_invalid_query_plan():
    retriever, _, _, _ = build_retriever()

    with pytest.raises(TypeError, match="query_plan must be a QueryPlan"):
        retriever.retrieve(None)


def make_test_evidence(
    chunk_id: str,
    relevance_score: float | None,
    query_id: str,
) -> Evidence:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content="Evidence content",
    )

    return Evidence(
        evidence_id=chunk_id,
        chunk=chunk,
        evidence_type=EvidenceType.PRIMARY,
        source_chunk_id=chunk_id,
        related_query_ids=[query_id],
        relevance_score=relevance_score,
    )


def test_merge_evidence_updates_missing_relevance_score():
    retriever, _, _, _ = build_retriever()

    evidence_by_chunk_id = {
        "chunk-1": make_test_evidence("chunk-1", None, "q1")
    }

    incoming = make_test_evidence("chunk-1", 0.8, "q2")

    retriever._merge_evidence(evidence_by_chunk_id, incoming)

    merged = evidence_by_chunk_id["chunk-1"]

    assert merged.relevance_score == 0.8
    assert set(merged.related_query_ids) == {"q1", "q2"}


def test_merge_evidence_keeps_highest_relevance_score():
    retriever, _, _, _ = build_retriever()

    evidence_by_chunk_id = {
        "chunk-1": make_test_evidence("chunk-1", 0.6, "q1")
    }

    incoming = make_test_evidence("chunk-1", 0.9, "q2")

    retriever._merge_evidence(evidence_by_chunk_id, incoming)

    merged = evidence_by_chunk_id["chunk-1"]

    assert merged.relevance_score == 0.9
    assert set(merged.related_query_ids) == {"q1", "q2"}
