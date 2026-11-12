import pytest

from cognexa.knowledge.retrieval.contracts import (
    CoverageReport,
    DocumentChunk,
    Evidence,
    EvidenceType,
    QueryPlan,
    QueryType,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStopReason,
    SufficiencyDecision,
)


def make_result():
    query = RetrievalQuery(
        query_id="q0",
        text="What is Python?",
        query_type=QueryType.ORIGINAL,
    )

    plan = QueryPlan(
        original_query="What is Python?",
        normalized_query="What is Python?",
        intent="definition",
        retrieval_queries=[query],
    )

    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="Python is a programming language.",
    )

    evidence = Evidence(
        evidence_id="evidence-001",
        chunk=chunk,
        evidence_type=EvidenceType.PRIMARY,
        source_chunk_id="chunk-001",
        related_query_ids=["q0"],
    )

    return RetrievalResult(
        query_plan=plan,
        evidence=[evidence],
        coverage=CoverageReport(overall_score=1.0),
        decision=SufficiencyDecision(
            sufficient=True,
            reason="All required evidence was found.",
        ),
        iterations=1,
        stop_reason=RetrievalStopReason.THRESHOLD_REACHED,
    )


def test_retrieval_result_creation():
    result = make_result()

    assert result.sufficient is True
    assert result.original_query == "What is Python?"
    assert result.iterations == 1


def test_retrieval_result_rejects_zero_iterations():
    result = make_result()
    result.iterations = 0

    with pytest.raises(ValueError):
        result.__post_init__()


def test_threshold_stop_requires_sufficient_decision():
    result = make_result()
    result.decision = SufficiencyDecision(
        sufficient=False,
        missing_query_ids=("q1",),
    )

    with pytest.raises(ValueError):
        result.__post_init__()


def test_retrieval_result_rejects_duplicate_evidence_ids():
    result = make_result()
    result.evidence.append(result.evidence[0])

    with pytest.raises(ValueError):
        result.__post_init__()