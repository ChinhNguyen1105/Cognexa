from types import SimpleNamespace

import pytest

from cognexa.assistant.answer_selector import AnswerSelector
from cognexa.verification import SentenceStatus


class FakeAnswerVerifier:
    """Mock AnswerVerifier for unit tests."""

    def __init__(self, statuses=None):
        self.statuses = statuses or {}
        self.calls = []

    def verify(self, answer, evidence):
        self.calls.append({
            "answer": answer,
            "evidence": evidence,
        })

        status = self.statuses.get(
            answer,
            SentenceStatus.SUPPORTED,
        )

        return SimpleNamespace(
            sentences=[
                SimpleNamespace(status=status)
            ]
        )


@pytest.fixture
def context():
    return (
        "Python is a high-level programming language. "
        "Python supports object-oriented programming. "
        "Python is widely used in artificial intelligence."
    )


def make_candidate(sentence, score):
    return {
        "sentence": sentence,
        "score": score,
        "index": 0,
    }


def test_select_returns_supported_sentences(context):
    high_level = "Python is a high-level programming language."
    oop = "Python supports object-oriented programming."

    verifier = FakeAnswerVerifier()

    selector = AnswerSelector(
        answer_verifier=verifier,
    )

    candidates = [
        make_candidate(high_level, 0.8),
        make_candidate(oop, 0.95),
    ]

    result = selector.select(
        candidates=candidates,
        context=context,
    )

    assert result == [oop, high_level]


def test_select_rejects_unsupported_sentences(context):
    sentence = "Python supports object-oriented programming."

    verifier = FakeAnswerVerifier({
        sentence: SentenceStatus.UNSUPPORTED,
    })

    selector = AnswerSelector(
        answer_verifier=verifier,
    )

    result = selector.select(
        candidates=[make_candidate(sentence, 0.95)],
        context=context,
    )

    assert result == []


def test_select_rejects_uncertain_sentences(context):
    sentence = "Python supports object-oriented programming."

    verifier = FakeAnswerVerifier({
        sentence: SentenceStatus.UNCERTAIN,
    })

    selector = AnswerSelector(
        answer_verifier=verifier,
    )

    result = selector.select(
        candidates=[make_candidate(sentence, 0.95)],
        context=context,
    )

    assert result == []


def test_select_does_not_filter_raw_score_by_default(context):
    sentence = "Python is a high-level programming language."

    selector = AnswerSelector(
        answer_verifier=FakeAnswerVerifier(),
    )

    result = selector.select(
        candidates=[make_candidate(sentence, -5.0)],
        context=context,
    )

    assert result == [sentence]


def test_select_respects_min_relevance_score(context):
    high_score = "Python supports object-oriented programming."
    low_score = "Python is a high-level programming language."

    selector = AnswerSelector(
        answer_verifier=FakeAnswerVerifier(),
        min_relevance_score=0.5,
    )

    candidates = [
        make_candidate(high_score, 0.8),
        make_candidate(low_score, 0.2),
    ]

    result = selector.select(
        candidates=candidates,
        context=context,
    )

    assert result == [high_score]


def test_select_removes_redundant_sentences(context):
    first = "Python supports object-oriented programming."
    second = "Python supports object-oriented programming very well."

    selector = AnswerSelector(
        answer_verifier=FakeAnswerVerifier(),
        redundancy_threshold=0.7,
    )

    result = selector.select(
        candidates=[
            make_candidate(first, 0.95),
            make_candidate(second, 0.9),
        ],
        context=context,
    )

    assert result == [first]


def test_select_respects_max_sentences(context):
    sentences = [
        "Python is a high-level programming language.",
        "Python supports object-oriented programming.",
        "Python is widely used in artificial intelligence.",
    ]

    selector = AnswerSelector(
        answer_verifier=FakeAnswerVerifier(),
        max_sentences=2,
    )

    candidates = [
        make_candidate(sentence, score)
        for sentence, score in zip(sentences, [0.9, 0.8, 0.7])
    ]

    result = selector.select(
        candidates=candidates,
        context=context,
    )

    assert len(result) == 2
    assert result == sentences[:2]


def test_select_returns_empty_for_empty_candidates(context):
    verifier = FakeAnswerVerifier()
    selector = AnswerSelector(answer_verifier=verifier)

    result = selector.select(
        candidates=[],
        context=context,
    )

    assert result == []
    assert verifier.calls == []


def test_select_returns_empty_for_empty_context():
    verifier = FakeAnswerVerifier()
    selector = AnswerSelector(answer_verifier=verifier)

    result = selector.select(
        candidates=[
            make_candidate("Python is a programming language.", 0.9)
        ],
        context="",
    )

    assert result == []
    assert verifier.calls == []


def test_init_rejects_non_positive_max_sentences():
    with pytest.raises(
        ValueError,
        match="max_sentences must be greater than zero",
    ):
        AnswerSelector(
            answer_verifier=FakeAnswerVerifier(),
            max_sentences=0,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_init_rejects_invalid_redundancy_threshold(threshold):
    with pytest.raises(
        ValueError,
        match="redundancy_threshold must be between 0 and 1",
    ):
        AnswerSelector(
            answer_verifier=FakeAnswerVerifier(),
            redundancy_threshold=threshold,
        )


def test_jaccard_similarity_returns_one_for_identical_sentences():
    sentence = "Python supports object oriented programming."

    similarity = AnswerSelector._jaccard_similarity(
        sentence,
        sentence,
    )

    assert similarity == 1.0


def test_jaccard_similarity_returns_zero_for_empty_tokens():
    similarity = AnswerSelector._jaccard_similarity(
        "",
        "",
    )

    assert similarity == 0.0


def test_select_detailed_prioritizes_distinct_aspects(context):
    first = "Python is a high-level programming language."
    second = "Python supports object-oriented programming."

    selector = AnswerSelector(
        FakeAnswerVerifier(),
        max_sentences=2,
    )

    result = selector.select_detailed([
        {
            "sentence": first,
            "score": 0.99,
            "question_score": 0.99,
            "aspect_scores": {"definition": 0.8},
            "covered_aspects": ["definition"],
        },
        {
            "sentence": second,
            "score": 0.2,
            "question_score": 0.2,
            "aspect_scores": {"paradigm": 0.95},
            "covered_aspects": ["paradigm"],
        },
    ], context)

    assert set(result.texts) == {first, second}
    assert result.uncovered_aspects == []


def test_select_detailed_marks_uncovered_aspect(context):
    first = "Python is a high-level programming language."

    selector = AnswerSelector(FakeAnswerVerifier())

    result = selector.select_detailed([
        {
            "sentence": first,
            "score": 0.9,
            "aspect_scores": {"definition": 0.9},
            "covered_aspects": ["definition"],
        },
        {
            "sentence": "Python supports object-oriented programming.",
            "score": 0.8,
            "aspect_scores": {"paradigm": 0.8},
        },
    ], context)

    assert result.uncovered_aspects == ["paradigm"]