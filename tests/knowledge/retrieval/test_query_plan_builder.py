import pytest

from cognexa.knowledge.retrieval.query_plan_builder import QueryPlanBuilder
from cognexa.knowledge.retrieval.contracts.query import QueryType


@pytest.fixture
def builder():
    return QueryPlanBuilder()


def test_build_simple_query_plan(builder):
    question = "What is Python?"

    plan = builder.build(
        original_question=question,
        decomposed_questions=[
            {
                "aspect": "original",
                "question": question,
            }
        ],
    )

    assert plan.original_query == question
    assert plan.normalized_query == "what is python?"
    assert plan.intent == "simple"
    assert len(plan.retrieval_queries) == 1
    assert plan.retrieval_queries[0].query_id == "original"
    assert plan.retrieval_queries[0].query_type == QueryType.ORIGINAL
    assert plan.sub_questions == []


def test_build_complex_query_plan(builder):
    question = "Compare RAG and fine-tuning."

    decomposed = [
        {
            "aspect": "comparison",
            "question": "How do RAG and fine-tuning differ?",
        },
        {
            "aspect": "comparison",
            "question": "How do RAG and fine-tuning differ in inference?",
        },
    ]

    plan = builder.build(question, decomposed)

    assert plan.intent == "comparison"
    assert len(plan.retrieval_queries) == 3
    assert len(plan.sub_questions) == 2

    assert plan.retrieval_queries[0].query_type == QueryType.ORIGINAL

    assert all(
        query.query_type == QueryType.SUB_QUESTION
        for query in plan.retrieval_queries[1:]
    )


def test_build_removes_duplicate_questions(builder):
    question = "What is RAG?"

    decomposed = [
        {
            "aspect": "definition",
            "question": "What is RAG?",
        },
        {
            "aspect": "process",
            "question": "How does RAG work?",
        },
        {
            "aspect": "process",
            "question": "How does RAG work?",
        },
    ]

    plan = builder.build(question, decomposed)

    assert len(plan.retrieval_queries) == 2
    assert plan.sub_questions == ["How does RAG work?"]


def test_build_preserves_original_question(builder):
    question = "What is machine learning?"

    plan = builder.build(
        question,
        [
            {
                "aspect": "definition",
                "question": "What is machine learning?",
            },
            {
                "aspect": "process",
                "question": "How does machine learning work?",
            },
        ],
    )

    assert plan.retrieval_queries[0].text == question
    assert plan.retrieval_queries[0].query_id == "original"


def test_build_with_empty_decomposition(builder):
    question = "What is Python?"

    plan = builder.build(question, [])

    assert len(plan.retrieval_queries) == 1
    assert plan.sub_questions == []
    assert plan.intent == "simple"


def test_build_rejects_invalid_original_question(builder):
    with pytest.raises(TypeError):
        builder.build(
            original_question=123,
            decomposed_questions=[],
        )

    with pytest.raises(ValueError):
        builder.build(
            original_question="   ",
            decomposed_questions=[],
        )


def test_build_rejects_invalid_decomposed_questions(builder):
    with pytest.raises(TypeError):
        builder.build("What is Python?", "invalid")

    with pytest.raises(TypeError):
        builder.build("What is Python?", ["invalid"])

    with pytest.raises(TypeError):
        builder.build(
            "What is Python?",
            [{"aspect": 123, "question": "What is Python?"}],
        )

    with pytest.raises(TypeError):
        builder.build(
            "What is Python?",
            [{"aspect": "definition", "question": 123}],
        )
