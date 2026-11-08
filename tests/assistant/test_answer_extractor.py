import pytest

from cognexa.assistant.answer_extractor import AnswerExtractor


class FakeReranker:
    """Mock CrossEncoder reranker for unit tests."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.last_query = None
        self.last_candidates = []

    def rerank(self, query, candidates):
        self.last_query = query
        self.last_candidates = candidates

        results = [
            {
                "index": index,
                "sentence": sentence,
                "score": self.scores.get(sentence, 0.0),
            }
            for index, sentence in enumerate(candidates)
        ]

        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )


@pytest.fixture
def context():
    return (
        "Python is a high-level programming language. "
        "Python supports object-oriented programming. "
        "Python is widely used in artificial intelligence."
    )


def test_extract_candidates_ranks_sentences_by_score(context):
    high_level = "Python is a high-level programming language."
    oop = "Python supports object-oriented programming."
    ai = "Python is widely used in artificial intelligence."

    reranker = FakeReranker(
        scores={
            high_level: 0.8,
            oop: 0.95,
            ai: 0.1,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=5,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="what paradigm does python support",
        context=context,
    )

    assert [item["sentence"] for item in result] == [
        oop,
        high_level,
        ai,
    ]

    assert [item["score"] for item in result] == [
        0.95,
        0.8,
        0.1,
    ]


def test_extract_candidates_passes_question_to_reranker(context):
    question = "what paradigm does python support"
    reranker = FakeReranker()

    extractor = AnswerExtractor(reranker=reranker)

    extractor.extract_candidates(
        question=question,
        context=context,
    )

    assert reranker.last_query == question


def test_extract_candidates_returns_empty_for_empty_context():
    reranker = FakeReranker()
    extractor = AnswerExtractor(reranker=reranker)

    result = extractor.extract_candidates(
        question="what is python",
        context="",
    )

    assert result == []
    assert reranker.last_query is None


def test_extract_candidates_rejects_empty_question():
    extractor = AnswerExtractor(reranker=FakeReranker())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        extractor.extract_candidates(
            question="",
            context="Python is a programming language.",
        )


def test_extract_candidates_rejects_whitespace_question():
    extractor = AnswerExtractor(reranker=FakeReranker())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        extractor.extract_candidates(
            question="   ",
            context="Python is a programming language.",
        )


def test_extract_candidates_respects_max_candidates(context):
    reranker = FakeReranker(
        scores={
            "Python is a high-level programming language.": 0.8,
            "Python supports object-oriented programming.": 0.95,
            "Python is widely used in artificial intelligence.": 0.7,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=2,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="what is python",
        context=context,
    )

    assert len(result) == 2


def test_extract_candidates_respects_score_gap_threshold(context):
    high_level = "Python is a high-level programming language."
    oop = "Python supports object-oriented programming."
    ai = "Python is widely used in artificial intelligence."

    reranker = FakeReranker(
        scores={
            high_level: 0.8,
            oop: 0.95,
            ai: -3.5,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="what paradigm does python support",
        context=context,
    )

    assert [item["sentence"] for item in result] == [
        oop,
        high_level,
    ]


def test_extract_candidates_keeps_minimum_candidates(context):
    reranker = FakeReranker(
        scores={
            "Python is a high-level programming language.": 0.8,
            "Python supports object-oriented programming.": 0.95,
            "Python is widely used in artificial intelligence.": -10.0,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=1.0,
    )

    result = extractor.extract_candidates(
        question="what is python",
        context=context,
    )

    assert len(result) >= 2


def test_split_sentences_removes_document_headers():
    text = (
        "[Document 1 | Source: python.txt | Chunk: 0]\n"
        "Python is a high-level programming language.\n"
        "Python supports object-oriented programming."
    )

    result = AnswerExtractor._split_sentences(text)

    assert result == [
        "Python is a high-level programming language.",
        "Python supports object-oriented programming.",
    ]


def test_split_sentences_removes_duplicates():
    text = (
        "Python is a programming language. "
        "Python is a programming language."
    )

    result = AnswerExtractor._split_sentences(text)

    assert result == [
        "Python is a programming language.",
    ]


def test_split_sentences_returns_empty_for_empty_text():
    result = AnswerExtractor._split_sentences("")

    assert result == []


def test_init_rejects_invalid_min_candidates():
    with pytest.raises(ValueError, match="min_candidates"):
        AnswerExtractor(
            reranker=FakeReranker(),
            min_candidates=0,
        )


def test_init_rejects_max_candidates_smaller_than_minimum():
    with pytest.raises(ValueError, match="max_candidates"):
        AnswerExtractor(
            reranker=FakeReranker(),
            min_candidates=3,
            max_candidates=2,
        )


def test_init_rejects_negative_score_gap_threshold():
    with pytest.raises(ValueError, match="score_gap_threshold"):
        AnswerExtractor(
            reranker=FakeReranker(),
            score_gap_threshold=-1.0,
        )


def test_extract_candidates_ranks_by_question_and_aspect(context):
    class MultiQueryReranker:
        def rerank(self, query, candidates):
            scores = {
                "main question": [0.9, 0.2, 0.1],
                "aspect second": [0.1, 0.95, 0.2],
            }[query]
            return [{"index": i, "sentence": sentence, "score": scores[i]} for i, sentence in enumerate(candidates)]

    extractor = AnswerExtractor(MultiQueryReranker(), min_candidates=1, max_candidates=5)
    result = extractor.extract_candidates("main question", context, ["aspect second"])
    second = "Python supports object-oriented programming."
    candidate = next(item for item in result if item["sentence"] == second)
    assert candidate["question_score"] == 0.2
    assert candidate["aspect_scores"]["aspect second"] == 0.95


def test_missing_aspect_score_is_not_fabricated(context):
    extractor = AnswerExtractor(FakeReranker(), min_candidates=1)
    result = extractor.extract_candidates("main", context, ["one aspect"])
    assert all("one aspect" in item["aspect_scores"] for item in result)
