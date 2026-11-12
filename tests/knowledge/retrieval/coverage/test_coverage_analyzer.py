import pytest

from cognexa.knowledge.retrieval.contracts import (
    DocumentChunk,
    Evidence,
    EvidenceType,
    QueryPlan,
    QueryType,
    RetrievalQuery,
)
from cognexa.knowledge.retrieval.coverage.coverage_analyzer import (
    CoverageAnalyzer,
)


@pytest.fixture
def analyzer():
    return CoverageAnalyzer()


def make_query_plan(
    queries: list[RetrievalQuery],
) -> QueryPlan:
    return QueryPlan(
        original_query="What is Python?",
        normalized_query="what is python",
        intent="information",
        retrieval_queries=queries,
    )


def make_query(
    query_id: str,
    text: str,
    weight: float = 1.0,
    query_type: QueryType = QueryType.SUB_QUESTION,
) -> RetrievalQuery:
    return RetrievalQuery(
        query_id=query_id,
        text=text,
        query_type=query_type,
        weight=weight,
    )


def make_evidence(
    evidence_id: str,
    related_query_ids: list[str],
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk=DocumentChunk(
            chunk_id=f"chunk-{evidence_id}",
            document_id="doc-1",
            content=f"Content for {evidence_id}",
        ),
        evidence_type=EvidenceType.PRIMARY,
        source_chunk_id=f"chunk-{evidence_id}",
        related_query_ids=related_query_ids,
    )


def test_all_queries_are_covered(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
        make_query("q2", "What are Python data types?"),
    ])

    evidence = [
        make_evidence("e1", ["q1"]),
        make_evidence("e2", ["q2"]),
    ]

    report = analyzer.evaluate(plan, evidence)

    assert report.overall_score == 1.0
    assert len(report.sub_question_results) == 2
    assert all(item.covered for item in report.sub_question_results)


def test_no_queries_are_covered(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
        make_query("q2", "What are Python data types?"),
    ])

    report = analyzer.evaluate(plan, [])

    assert report.overall_score == 0.0
    assert all(not item.covered for item in report.sub_question_results)


def test_partial_coverage(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
        make_query("q2", "What are Python data types?"),
        make_query("q3", "What are Python decorators?"),
    ])

    evidence = [
        make_evidence("e1", ["q1"]),
        make_evidence("e2", ["q2"]),
    ]

    report = analyzer.evaluate(plan, evidence)

    assert report.overall_score == pytest.approx(2 / 3)

    results = {
        item.query_id: item
        for item in report.sub_question_results
    }

    assert results["q1"].covered is True
    assert results["q2"].covered is True
    assert results["q3"].covered is False


def test_weighted_coverage(analyzer):
    plan = make_query_plan([
        make_query(
            "q1",
            "What is Python?",
            weight=3.0,
            query_type=QueryType.ORIGINAL,
        ),
        make_query("q2", "What are Python data types?", weight=1.0),
    ])

    evidence = [
        make_evidence("e1", ["q1"]),
    ]

    report = analyzer.evaluate(plan, evidence)

    assert report.overall_score == pytest.approx(0.75)


def test_multiple_evidence_items_for_same_query(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
    ])

    evidence = [
        make_evidence("e1", ["q1"]),
        make_evidence("e2", ["q1"]),
        make_evidence("e3", ["q1"]),
    ]

    report = analyzer.evaluate(plan, evidence)

    result = report.sub_question_results[0]

    assert report.overall_score == 1.0
    assert result.covered is True
    assert result.evidence_ids == ["e1", "e2", "e3"]


def test_unrelated_evidence_does_not_count(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
    ])

    evidence = [
        make_evidence("e1", ["another-query"]),
    ]

    report = analyzer.evaluate(plan, evidence)

    assert report.overall_score == 0.0
    assert report.sub_question_results[0].covered is False
    assert report.sub_question_results[0].evidence_ids == []


def test_missing_information_is_recorded(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
    ])

    report = analyzer.evaluate(plan, [])

    result = report.sub_question_results[0]

    assert result.missing_information is not None
    assert "No evidence" in result.missing_information


def test_invalid_query_plan_type(analyzer):
    with pytest.raises(TypeError, match="query_plan"):
        analyzer.evaluate("invalid", [])


def test_invalid_evidence_item_type(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
    ])

    with pytest.raises(TypeError, match="Evidence"):
        analyzer.evaluate(plan, ["invalid"])


def test_string_is_not_accepted_as_evidence_sequence(analyzer):
    plan = make_query_plan([
        make_query("q1", "What is Python?", query_type=QueryType.ORIGINAL),
    ])

    with pytest.raises(TypeError, match="sequence"):
        analyzer.evaluate(plan, "invalid")
