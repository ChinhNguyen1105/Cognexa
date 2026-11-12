import pytest

from cognexa.knowledge.retrieval.contracts import (
    CoverageReport,
    DocumentChunk,
    Evidence,
    EvidenceType,
    SubQuestionCoverage,
    SufficiencyDecision,
)


def make_evidence():
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="Python is a programming language.",
    )

    return Evidence(
        evidence_id="evidence-001",
        chunk=chunk,
        evidence_type=EvidenceType.PRIMARY,
        source_chunk_id="chunk-001",
        related_query_ids=["q1"],
        relevance_score=0.95,
    )


def test_evidence_creation():
    evidence = make_evidence()

    assert evidence.evidence_type == EvidenceType.PRIMARY
    assert evidence.source_chunk_id == "chunk-001"
    assert evidence.relevance_score == 0.95


def test_sub_question_coverage_creation():
    result = SubQuestionCoverage(
        query_id="q1",
        question="What is Python?",
        coverage_score=0.95,
        covered=True,
        evidence_ids=["evidence-001"],
    )

    assert result.covered is True
    assert result.coverage_score == 0.95


def test_covered_question_requires_evidence():
    with pytest.raises(ValueError):
        SubQuestionCoverage(
            query_id="q1",
            question="What is Python?",
            coverage_score=0.95,
            covered=True,
        )


def test_coverage_score_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        SubQuestionCoverage(
            query_id="q1",
            question="What is Python?",
            coverage_score=1.5,
            covered=False,
        )


def test_coverage_report_creation():
    result = SubQuestionCoverage(
        query_id="q1",
        question="What is Python?",
        coverage_score=0.95,
        covered=True,
        evidence_ids=["evidence-001"],
    )

    report = CoverageReport(
        overall_score=0.95,
        sub_question_results=[result],
    )

    assert report.overall_score == 0.95
    assert len(report.sub_question_results) == 1


def test_coverage_report_rejects_duplicate_query_ids():
    result1 = SubQuestionCoverage(
        query_id="q1",
        question="What is Python?",
        coverage_score=0.95,
        covered=True,
        evidence_ids=["e1"],
    )

    result2 = SubQuestionCoverage(
        query_id="q1",
        question="Explain Python",
        coverage_score=0.80,
        covered=True,
        evidence_ids=["e2"],
    )

    with pytest.raises(ValueError):
        CoverageReport(
            overall_score=0.90,
            sub_question_results=[result1, result2],
        )


def test_sufficiency_decision_creation():
    decision = SufficiencyDecision(
        sufficient=False,
        missing_query_ids=("q2",),
        reason="Missing evidence for comparison.",
    )

    assert decision.sufficient is False
    assert decision.missing_query_ids == ("q2",)


def test_sufficient_decision_cannot_have_missing_queries():
    with pytest.raises(ValueError):
        SufficiencyDecision(
            sufficient=True,
            missing_query_ids=("q2",),
        )