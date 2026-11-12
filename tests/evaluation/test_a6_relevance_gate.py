from types import SimpleNamespace

from cognexa.assistant.answerability_checker import AnswerabilityChecker
from cognexa.assistant.answer_extractor import AnswerExtractor


class FakeReranker:
    def rerank(self, question, sentences):
        scores = {
            "GPU memory usage is not discussed here.": -1.9,
            "Overfitting reduces generalization ability.": -3.0,
            "A model performs poorly on unseen data.": -3.8,
            "A relevant sentence about training data.": 2.5,
        }

        return [
            {
                "index": index,
                "sentence": sentence,
                "score": scores[sentence],
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
