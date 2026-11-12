import pytest

from cognexa.knowledge.retrieval.contracts import (
    CoverageReport,
    SubQuestionCoverage,
)
from cognexa.knowledge.retrieval.coverage.sufficiency_checker import (
    SufficiencyChecker,
)


def make_report(
    overall_score: float,
    covered_ids: list[str],
    uncovered_ids: list[str],
    optional_ids: list[str] | None = None,
) -> CoverageReport:
    optional_ids = set(optional_ids or [])

    results = [
        SubQuestionCoverage(
            query_id=query_id,
            question=f"Question {query_id}",
            coverage_score=1.0,
            covered=True,
            evidence_ids=[f"evidence-{query_id}"],
            required=query_id not in optional_ids,
        )
        for query_id in covered_ids
    ]

    results.extend(
        SubQuestionCoverage(
            query_id=query_id,
            question=f"Question {query_id}",
            coverage_score=0.0,
            covered=False,
            required=query_id not in optional_ids,
        )
        for query_id in uncovered_ids
    )

    return CoverageReport(
        overall_score=overall_score,
        sub_question_results=results,
    )


def test_sufficient_when_coverage_exceeds_threshold():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(
        0.8,
        ["q1", "q2"],
        ["q3"],
        optional_ids=["q3"],
    )

    decision = checker.check(report)

    assert decision.sufficient is True
    assert decision.missing_query_ids == ()
    assert "meets" in decision.reason


def test_insufficient_when_coverage_below_threshold():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(0.5, ["q1"], ["q2", "q3"])

    decision = checker.check(report)

    assert decision.sufficient is False
    assert decision.missing_query_ids == ("q2", "q3")
    assert "below" in decision.reason


def test_sufficient_when_coverage_equals_threshold():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(
        0.7,
        ["q1", "q2"],
        ["q3"],
        optional_ids=["q3"],
    )

    decision = checker.check(report)

    assert decision.sufficient is True
    assert decision.missing_query_ids == ()


def test_required_uncovered_query_blocks_sufficiency():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(0.9, ["q1", "q2"], ["q3"])

    decision = checker.check(report)

    assert decision.sufficient is False
    assert decision.missing_query_ids == ("q3",)
    assert "q3" in decision.reason
    assert "Required queries" in decision.reason


def test_optional_uncovered_query_does_not_block_sufficiency():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(
        0.9,
        ["q1", "q2"],
        ["q3"],
        optional_ids=["q3"],
    )

    decision = checker.check(report)

    assert decision.sufficient is True
    assert decision.missing_query_ids == ()


def test_all_queries_uncovered():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(0.0, [], ["q1", "q2"])

    decision = checker.check(report)

    assert decision.sufficient is False
    assert decision.missing_query_ids == ("q1", "q2")


def test_all_queries_covered():
    checker = SufficiencyChecker(min_coverage=0.7)
    report = make_report(1.0, ["q1", "q2"], [])

    decision = checker.check(report)

    assert decision.sufficient is True
    assert decision.missing_query_ids == ()


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_min_coverage_value(value):
    with pytest.raises(ValueError):
        SufficiencyChecker(min_coverage=value)


@pytest.mark.parametrize("value", ["0.7", None, True])
def test_invalid_min_coverage_type(value):
    with pytest.raises(TypeError):
        SufficiencyChecker(min_coverage=value)


def test_invalid_report_type():
    checker = SufficiencyChecker()

    with pytest.raises(TypeError, match="CoverageReport"):
        checker.check("invalid")


def test_zero_threshold():
    checker = SufficiencyChecker(min_coverage=0.0)
    report = make_report(
        0.0,
        [],
        ["q1"],
        optional_ids=["q1"],
    )

    decision = checker.check(report)

    assert decision.sufficient is True
    assert decision.missing_query_ids == ()


def test_one_threshold():
    checker = SufficiencyChecker(min_coverage=1.0)
    report = make_report(0.9, ["q1"], ["q2"])

    decision = checker.check(report)

    assert decision.sufficient is False
    assert decision.missing_query_ids == ("q2",)
