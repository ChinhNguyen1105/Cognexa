
import math

import pytest

from cognexa.knowledge.evaluation.metrics.answer_relevance import (
    calculate_answer_question_relevance,
)


class MockReranker:
    def __init__(self, results=None):
        self.results = (
            results if results is not None else [{"score": 2.5}]
        )
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append({
            "query": query,
            "candidates": candidates,
        })
        return self.results


def test_calculates_answer_question_relevance():
    reranker = MockReranker(results=[{"score": 3.25}])

    score = calculate_answer_question_relevance(
        question="How can overfitting be reduced?",
        answer="Regularization can reduce overfitting.",
        reranker=reranker,
    )

    assert score == 3.25
    assert reranker.calls == [
        {
            "query": "How can overfitting be reduced?",
            "candidates": [
                "Regularization can reduce overfitting."
            ],
        }
    ]


def test_returns_none_for_empty_answer():
    reranker = MockReranker()

    score = calculate_answer_question_relevance(
        question="What is overfitting?",
        answer="   ",
        reranker=reranker,
    )

    assert score is None
    assert reranker.calls == []


def test_rejects_empty_question():
    reranker = MockReranker()

    with pytest.raises(ValueError, match="Question"):
        calculate_answer_question_relevance(
            question=" ",
            answer="Overfitting occurs when a model memorizes training data.",
            reranker=reranker,
        )

    assert reranker.calls == []


def test_rejects_non_string_question():
    reranker = MockReranker()

    with pytest.raises(ValueError, match="Question"):
        calculate_answer_question_relevance(
            question=None,
            answer="Some answer.",
            reranker=reranker,
        )


def test_rejects_non_string_answer():
    reranker = MockReranker()

    with pytest.raises(TypeError, match="Answer"):
        calculate_answer_question_relevance(
            question="What is overfitting?",
            answer=None,
            reranker=reranker,
        )


def test_rejects_object_without_rerank_method():
    with pytest.raises(TypeError, match="rerank"):
        calculate_answer_question_relevance(
            question="What is overfitting?",
            answer="Overfitting is a generalization problem.",
            reranker=object(),
        )


def test_rejects_empty_reranker_results():
    reranker = MockReranker(results=[])

    with pytest.raises(RuntimeError, match="no relevance score"):
        calculate_answer_question_relevance(
            question="What is overfitting?",
            answer="Overfitting is a generalization problem.",
            reranker=reranker,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        None,
        "high",
        True,
    ],
)
def test_rejects_invalid_relevance_score(invalid_score):
    reranker = MockReranker(
        results=[{"score": invalid_score}]
    )

    with pytest.raises(RuntimeError, match="invalid relevance score"):
        calculate_answer_question_relevance(
            question="What is overfitting?",
            answer="Overfitting is a generalization problem.",
            reranker=reranker,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_non_finite_relevance_score(invalid_score):
    reranker = MockReranker(
        results=[{"score": invalid_score}]
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        calculate_answer_question_relevance(
            question="What is overfitting?",
            answer="Overfitting is a generalization problem.",
            reranker=reranker,
        )


def test_preserves_negative_raw_score():
    reranker = MockReranker(results=[{"score": -1.75}])

    score = calculate_answer_question_relevance(
        question="How can overfitting be reduced?",
        answer="The sky is blue.",
        reranker=reranker,
    )

    assert score == -1.75

