import pytest

from cognexa.knowledge.retrieval.contracts import (
    QueryPlan,
    QueryType,
    RetrievalQuery,
)


def make_query_plan():
    original = RetrievalQuery(
        query_id="q0",
        text="Explain overfitting and underfitting",
        query_type=QueryType.ORIGINAL,
    )

    sub_question = RetrievalQuery(
        query_id="q1",
        text="What is overfitting?",
        query_type=QueryType.SUB_QUESTION,
    )

    return QueryPlan(
        original_query="Explain overfitting and underfitting",
        normalized_query="Explain overfitting and underfitting",
        intent="conceptual_comparison",
        retrieval_queries=[original, sub_question],
        sub_questions=["What is overfitting?"],
    )


def test_retrieval_query_creation():
    query = RetrievalQuery(
        query_id="q1",
        text="What is Python?",
    )

    assert query.query_id == "q1"
    assert query.query_type == QueryType.ORIGINAL
    assert query.weight == 1.0


def test_retrieval_query_rejects_empty_text():
    with pytest.raises(ValueError):
        RetrievalQuery(query_id="q1", text=" ")


def test_retrieval_query_rejects_non_positive_weight():
    with pytest.raises(ValueError):
        RetrievalQuery(
            query_id="q1",
            text="What is Python?",
            weight=0,
        )


def test_query_plan_creation():
    plan = make_query_plan()

    assert plan.original_query
    assert len(plan.retrieval_queries) == 2
    assert len(plan.sub_questions) == 1


def test_query_plan_requires_original_query():
    query = RetrievalQuery(
        query_id="q1",
        text="What is Python?",
        query_type=QueryType.SUB_QUESTION,
    )

    with pytest.raises(ValueError):
        QueryPlan(
            original_query="What is Python?",
            normalized_query="What is Python?",
            intent="definition",
            retrieval_queries=[query],
        )


def test_query_plan_rejects_duplicate_query_ids():
    query1 = RetrievalQuery(
        query_id="q1",
        text="What is Python?",
        query_type=QueryType.ORIGINAL,
    )

    query2 = RetrievalQuery(
        query_id="q1",
        text="Explain Python",
        query_type=QueryType.SUB_QUESTION,
    )

    with pytest.raises(ValueError):
        QueryPlan(
            original_query="What is Python?",
            normalized_query="What is Python?",
            intent="definition",
            retrieval_queries=[query1, query2],
        )


def test_query_plan_requires_retrieval_queries():
    with pytest.raises(ValueError):
        QueryPlan(
            original_query="What is Python?",
            normalized_query="What is Python?",
            intent="definition",
        )