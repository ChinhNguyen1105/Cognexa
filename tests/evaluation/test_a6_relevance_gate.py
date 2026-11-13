from types import SimpleNamespace

from cognexa.assistant.answerability_checker import AnswerabilityChecker
from cognexa.assistant.answer_extractor import AnswerExtractor


class FakeReranker:
    def __init__(self, scores=None):
        if scores is None:
            scores = {
                "GPU memory usage is not discussed here.": -1.9,
                "Overfitting reduces generalization ability.": -3.0,
                "A model performs poorly on unseen data.": -3.8,
                "A relevant sentence about training data.": 2.5,
            }
        self.scores = scores

    def rerank(self, question, sentences):
        return [
            {
                "index": index,
                "sentence": sentence,
                "score": self.scores[sentence],
            }
            for index, sentence in enumerate(sentences)
        ]


def test_answerability_rejects_negative_relevance():
    checker = AnswerabilityChecker()

    selection = SimpleNamespace(
        sentences=[
            "Overfitting reduces generalization ability."
        ],
        uncovered_aspects=[],
        aspects=[],
        rejected=[],
        top_relevance=-1.926,
    )

    result = checker.assess(
        "How does overfitting affect GPU memory consumption?",
        selection,
    )

    assert result.status == "INSUFFICIENT"
    assert result.answerable is False


def test_answerability_accepts_relevant_verified_evidence():
    checker = AnswerabilityChecker()

    selection = SimpleNamespace(
        sentences=[
            "A relevant sentence about training data."
        ],
        uncovered_aspects=[],
        aspects=[],
        rejected=[],
        top_relevance=2.5,
    )

    result = checker.assess(
        "What is the effect of training data?",
        selection,
    )

    assert result.status == "SUFFICIENT"
    assert result.answerable is True


def test_extractor_does_not_force_negative_candidates():
    extractor = AnswerExtractor(
        reranker=FakeReranker(),
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=3.0,
        min_score=0.0,
    )

    candidates = extractor.extract_candidates(
        question="How does overfitting affect GPU memory?",
        context=(
            "GPU memory usage is not discussed here. "
            "Overfitting reduces generalization ability. "
            "A model performs poorly on unseen data."
        ),
    )

    assert candidates == []


def test_extractor_keeps_candidates_above_absolute_threshold():
    extractor = AnswerExtractor(
        reranker=FakeReranker(),
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=3.0,
        min_score=0.0,
    )

    candidates = extractor.extract_candidates(
        question="What is relevant to training data?",
        context=(
            "A relevant sentence about training data. "
            "Overfitting reduces generalization ability. "
            "A model performs poorly on unseen data."
        ),
    )

    assert len(candidates) == 1
    assert candidates[0]["score"] == 2.5


def test_negative_raw_logits_still_rank_relevant_candidates():
    reranker = FakeReranker(
        scores={
            "A relevant sentence about training data.": -0.75,
            "Overfitting reduces generalization ability.": -3.1,
            "A model performs poorly on unseen data.": -4.0,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=1,
        max_candidates=10,
        score_gap_threshold=3.0,
        min_score=None,
    )

    candidates = extractor.extract_candidates(
        question="What is the effect of training data?",
        context=(
            "A relevant sentence about training data. "
            "Overfitting reduces generalization ability. "
            "A model performs poorly on unseen data."
        ),
    )

    assert len(candidates) == 3
    assert [item["sentence"] for item in candidates] == [
        "A relevant sentence about training data.",
        "A model performs poorly on unseen data.",
        "Overfitting reduces generalization ability.",
    ]
    assert all(item["raw_score"] < 0.0 for item in candidates)


def test_positive_irrelevant_scores_are_not_treated_as_valid_by_sign():
    reranker = FakeReranker(
        scores={
            "General wording with no answer.": 1.0,
            "A specific answer sentence.": 0.9,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=1,
        max_candidates=10,
        score_gap_threshold=3.0,
        min_score=None,
    )

    candidates = extractor.extract_candidates(
        question="What is the effect of training data?",
        context=(
            "General wording with no answer. "
            "A specific answer sentence."
        ),
    )

    assert len(candidates) == 2
    assert [item["sentence"] for item in candidates] == [
        "General wording with no answer.",
        "A specific answer sentence.",
    ]
    assert all(item["raw_score"] >= 0.0 for item in candidates)


def test_extractor_keeps_negative_candidates_when_min_score_is_none():
    reranker = FakeReranker(
        scores={
            "A valid but negative score sentence.": -0.25,
            "Unrelated sentence.": -1.5,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=1,
        max_candidates=10,
        score_gap_threshold=0.0,
        min_score=None,
    )

    candidates = extractor.extract_candidates(
        question="What is the valid sentence?",
        context=(
            "A valid but negative score sentence. "
            "Unrelated sentence."
        ),
    )

    assert len(candidates) == 2
    assert [item["sentence"] for item in candidates] == [
        "A valid but negative score sentence.",
        "Unrelated sentence.",
    ]
    assert all(item["raw_score"] < 0.0 for item in candidates)


def test_answerability_allows_negative_raw_logits_when_selection_is_relevant():
    checker = AnswerabilityChecker(min_top_relevance=None)

    selection = SimpleNamespace(
        sentences=[{"sentence": "A relevant sentence about training data."}],
        uncovered_aspects=[],
        aspects=[],
        rejected=[],
        top_relevance=-0.75,
    )

    result = checker.assess(
        "What is the effect of training data?",
        selection,
    )

    assert result.status == "SUFFICIENT"
    assert result.answerable is True
