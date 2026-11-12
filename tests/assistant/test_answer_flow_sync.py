"""
Tests that keep AnswerExtractor, AnswerSelector, AnswerabilityChecker,
the verification package and Assistant consistent with each other.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cognexa.verification.verifier as verifier_module
from cognexa.assistant.answer_extractor import AnswerExtractor
from cognexa.assistant.answer_selector import (
    AnswerSelector,
    SelectionResult,
)
from cognexa.assistant.answerability_checker import AnswerabilityChecker
from cognexa.assistant.assistant import Assistant
from cognexa.verification import (
    AnswerVerifier,
    NLIScores,
    SentenceStatus,
)


# =========================
# Fakes
# =========================

class QueryAwareReranker:
    """Reranker whose scores depend on (query, sentence)."""

    def __init__(self, scores, default=0.0):
        self.scores = scores
        self.default = default

    def rerank(self, query, candidates):
        results = [
            {
                "index": index,
                "sentence": sentence,
                "score": self.scores.get(query, {}).get(
                    sentence, self.default
                ),
            }
            for index, sentence in enumerate(candidates)
        ]

        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )


class RecordingVerifier:
    """AnswerVerifier stand-in recording what the selector sends."""

    def __init__(self, statuses=None):
        self.statuses = statuses or {}
        self.calls = []

    def verify(self, answer, evidence):
        self.calls.append({"answer": answer, "evidence": evidence})

        status = self.statuses.get(answer, SentenceStatus.SUPPORTED)

        return SimpleNamespace(
            sentences=[SimpleNamespace(status=status)]
        )


class FakeNLIModel:
    def __init__(self, entails=True):
        self.entails = entails
        self.premises = []

    def predict_batch(self, premises, hypotheses):
        self.premises.extend(premises)

        scores = (
            NLIScores(0.95, 0.02, 0.03)
            if self.entails
            else NLIScores(0.05, 0.05, 0.90)
        )

        return [scores for _ in premises]


def candidate(sentence, score, aspect=None, query=None):
    item = {
        "sentence": sentence,
        "score": score,
        "question_score": score,
    }

    if aspect is not None:
        item["aspect_scores"] = {query or f"q-{aspect}": 0.9}
        item["aspect_labels"] = {query or f"q-{aspect}": aspect}
        item["covered_aspects"] = [aspect]

    return item


CONTEXT = (
    "[Document 1 | Source: python.txt | Chunk: 0]\n"
    "Python is a programming language. "
    "Python supports object-oriented programming.\n"
    "[Document 2 | Source: python.txt | Chunk: 1]\n"
    "Python is widely used in artificial intelligence."
)


# =========================
# AnswerExtractor -> covered_aspects
# =========================

def test_extractor_marks_best_sentences_as_covering_their_aspect():
    cause = "Too many epochs cause overfitting."
    effect = "Overfitting hurts generalization."
    other = "The weather is sunny."

    reranker = QueryAwareReranker({
        "What is overfitting?": {cause: 0.9, effect: 0.8, other: 0.1},
        "What causes it?": {cause: 0.95, effect: 0.2, other: 0.0},
        "What are its effects?": {effect: 0.95, cause: 0.2, other: 0.0},
    })

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=1,
        max_candidates=10,
        score_gap_threshold=100.0,
    )

    result = extractor.extract_candidates(
        question="What is overfitting?",
        context=f"{cause} {effect} {other}",
        aspect_queries=[
            {"aspect": "causes", "question": "What causes it?"},
            {"aspect": "effects", "question": "What are its effects?"},
        ],
    )

    by_sentence = {item["sentence"]: item for item in result}

    assert by_sentence[cause]["covered_aspects"] == ["causes"]
    assert by_sentence[effect]["covered_aspects"] == ["effects"]
    assert by_sentence[other]["covered_aspects"] == []


def test_extractor_without_aspect_queries_has_empty_coverage():
    reranker = QueryAwareReranker({})
    extractor = AnswerExtractor(reranker=reranker)

    result = extractor.extract_candidates(
        question="What is Python?",
        context="Python is a language. Python is popular.",
    )

    assert result
    assert all(item["covered_aspects"] == [] for item in result)


# =========================
# AnswerSelector
# =========================

def test_selector_verifies_against_clean_evidence_units():
    verifier = RecordingVerifier()
    selector = AnswerSelector(answer_verifier=verifier)

    selector.select_detailed(
        [candidate("Python is a programming language.", 0.9)],
        CONTEXT,
    )

    assert verifier.calls[0]["evidence"] == [
        "Python is a programming language.",
        "Python supports object-oriented programming.",
        "Python is widely used in artificial intelligence.",
    ]


def test_selector_returns_empty_when_context_only_has_headers():
    verifier = RecordingVerifier()
    selector = AnswerSelector(answer_verifier=verifier)

    result = selector.select_detailed(
        [candidate("Python is a language.", 0.9)],
        "[Document 1 | Source: python.txt]",
    )

    assert result.sentences == []
    assert verifier.calls == []


def test_selector_requires_every_verified_sentence_to_be_supported():
    class TwoSentenceVerifier:
        def verify(self, answer, evidence):
            return SimpleNamespace(sentences=[
                SimpleNamespace(status=SentenceStatus.SUPPORTED),
                SimpleNamespace(status=SentenceStatus.CONTRADICTED),
            ])

    selector = AnswerSelector(answer_verifier=TwoSentenceVerifier())

    result = selector.select_detailed(
        [candidate("Python is a programming language.", 0.9)],
        CONTEXT,
    )

    assert result.sentences == []
    assert result.rejected[0]["status"] == "contradicted"


def test_selector_reports_rejected_candidates_and_top_relevance():
    good = "Python is a programming language."
    bad = "Python supports object-oriented programming."

    selector = AnswerSelector(
        answer_verifier=RecordingVerifier({
            bad: SentenceStatus.UNCERTAIN,
        }),
    )

    result = selector.select_detailed(
        [candidate(bad, 0.95), candidate(good, 0.7)],
        CONTEXT,
    )

    assert result.texts == [good]
    assert result.top_relevance == 0.7
    assert result.rejected == [{
        "sentence": bad,
        "status": "uncertain",
        "relevance": 0.95,
    }]


def test_selector_reserves_slots_for_uncovered_aspects():
    a1 = "Alpha topic first fact is documented here."
    a2 = "Alpha topic second fact is documented there."
    a3 = "Alpha topic third fact appears elsewhere."
    b1 = "Beta topic fact lives in another place."

    context = " ".join([a1, a2, a3, b1])

    selector = AnswerSelector(
        answer_verifier=RecordingVerifier(),
        max_sentences=2,
    )

    result = selector.select_detailed(
        [
            candidate(a1, 0.99, "alpha"),
            candidate(a2, 0.98, "alpha"),
            candidate(a3, 0.97, "alpha"),
            candidate(b1, 0.10, "beta"),
        ],
        context,
    )

    assert b1 in result.texts
    assert len(result.texts) == 2
    assert result.uncovered_aspects == []
    # Output stays in relevance order.
    assert result.texts == [a1, b1]


def test_selector_falls_back_to_next_candidate_for_an_aspect():
    a1 = "Alpha topic first fact is documented here."
    b1 = "Beta topic best fact is rejected by verification."
    b2 = "Beta topic second fact is verified instead."

    selector = AnswerSelector(
        answer_verifier=RecordingVerifier({
            b1: SentenceStatus.UNSUPPORTED,
        }),
        max_sentences=3,
    )

    result = selector.select_detailed(
        [
            candidate(a1, 0.9, "alpha"),
            candidate(b1, 0.5, "beta"),
            candidate(b2, 0.4, "beta"),
        ],
        " ".join([a1, b1, b2]),
    )

    assert result.texts == [a1, b2]
    assert result.uncovered_aspects == []


def test_selector_does_not_verify_a_candidate_twice():
    a1 = "Alpha topic first fact is documented here."

    verifier = RecordingVerifier()
    selector = AnswerSelector(answer_verifier=verifier)

    selector.select_detailed(
        [candidate(a1, 0.9, "alpha")],
        a1,
    )

    assert len(verifier.calls) == 1


# =========================
# AnswerabilityChecker
# =========================

def selection(sentences=("A verified fact.",), **kwargs):
    # A6: relevance is an explicit input to AnswerabilityChecker.
    # Keep ordinary test selections relevant unless a test overrides it.
    kwargs.setdefault("top_relevance", 1.0)

    return SelectionResult(
        sentences=[{"sentence": text} for text in sentences],
        **kwargs,
    )


def test_checker_refuses_when_no_aspect_is_covered():
    result = AnswerabilityChecker().assess(
        "Question?",
        selection(
            aspects=["causes", "effects"],
            uncovered_aspects=["causes", "effects"],
        ),
    )

    assert result.status == "INSUFFICIENT"
    assert not result.answerable


def test_checker_keeps_partial_answers_when_some_aspect_is_covered():
    result = AnswerabilityChecker().assess(
        "Question?",
        selection(
            aspects=["causes", "effects"],
            uncovered_aspects=["effects"],
        ),
    )

    assert result.status == "PARTIAL"
    assert result.metrics["aspect_coverage"] == 0.5
    assert result.metrics["uncovered_aspects"] == ["effects"]


def test_checker_min_aspect_coverage_can_reject_partial_answers():
    result = AnswerabilityChecker(min_aspect_coverage=0.6).assess(
        "Question?",
        selection(
            aspects=["causes", "effects"],
            uncovered_aspects=["effects"],
        ),
    )

    assert result.status == "INSUFFICIENT"


def test_checker_refuses_below_min_top_relevance():
    checker = AnswerabilityChecker(min_top_relevance=2.0)

    refused = checker.assess("Question?", selection(top_relevance=1.5))
    answered = checker.assess("Question?", selection(top_relevance=2.5))

    assert refused.status == "INSUFFICIENT"
    assert answered.status == "SUFFICIENT"


def test_checker_refuses_when_relevance_is_unknown():
    checker = AnswerabilityChecker(min_top_relevance=2.0)

    result = checker.assess("Question?", selection(top_relevance=None))

    assert result.status == "INSUFFICIENT"
    assert not result.answerable


def test_checker_relevance_check_can_be_disabled():
    result = AnswerabilityChecker(min_top_relevance=None).assess(
        "Question?",
        selection(top_relevance=-10.0),
    )

    assert result.status == "SUFFICIENT"


def test_checker_reports_rejected_count():
    result = AnswerabilityChecker().assess(
        "Question?",
        selection(rejected=[{"sentence": "x", "status": "uncertain"}]),
    )

    assert result.metrics["rejected_count"] == 1


def test_checker_allow_partial_false_rejects_partial_answers():
    result = AnswerabilityChecker(allow_partial=False).assess(
        "Question?",
        selection(
            aspects=["causes", "effects"],
            uncovered_aspects=["effects"],
        ),
    )

    assert result.status == "INSUFFICIENT"


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_checker_rejects_invalid_min_aspect_coverage(value):
    with pytest.raises(ValueError):
        AnswerabilityChecker(min_aspect_coverage=value)


# =========================
# Assistant wiring
# =========================

class SingleQueryDecomposer:
    def decompose(self, question):
        return [question]


def build_assistant(
    *,
    entails=True,
    rerank_scores=None,
    answerability_checker=None,
):
    nli = FakeNLIModel(entails=entails)
    verifier = AnswerVerifier(nli_model=nli)

    reranker = QueryAwareReranker(rerank_scores or {})

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=100.0,
    )

    knowledge_base = MagicMock()
    knowledge_base.search.return_value = [{
        "id": "doc-1",
        "content": (
            "Python is a programming language. "
            "Python supports object-oriented programming."
        ),
        "metadata": {"file_name": "python.txt"},
        "distance": 0.1,
    }]
    knowledge_base.expand_to_parents.side_effect = lambda docs: docs

    context_validator = MagicMock()
    context_validator.is_relevant.return_value = True

    assistant = Assistant(
        generator=MagicMock(),
        prompt_builder=MagicMock(),
        knowledge_base=knowledge_base,
        context_validator=context_validator,
        answer_verifier=verifier,
        answer_extractor=extractor,
        answerability_checker=answerability_checker,
        query_decomposer=SingleQueryDecomposer(),
    )

    return assistant, nli


def test_assistant_default_verifier_is_the_verification_package_class():
    assistant = Assistant(
        generator=MagicMock(),
        prompt_builder=MagicMock(),
        knowledge_base=MagicMock(),
        answer_extractor=MagicMock(),
    )

    assert isinstance(assistant.answer_verifier, AnswerVerifier)
    assert isinstance(assistant.answer_selector, AnswerSelector)
    assert assistant.answer_selector.answer_verifier is (
        assistant.answer_verifier
    )
    assert callable(assistant.answer_verifier.verify)


def test_assistant_default_verifier_does_not_load_nli_model(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("NLIModel must be loaded lazily.")

    monkeypatch.setattr(verifier_module, "NLIModel", fail)

    Assistant(
        generator=MagicMock(),
        prompt_builder=MagicMock(),
        knowledge_base=MagicMock(),
        answer_extractor=MagicMock(),
    )


def test_assistant_answers_with_real_selector_and_verifier():
    assistant, nli = build_assistant(
        rerank_scores={
            "What is Python?": {
                "Python is a programming language.": 5.0,
                "Python supports object-oriented programming.": 3.0,
            }
        }
    )

    response = assistant.chat("What is Python?")

    assert response == (
        "Python is a programming language. "
        "Python supports object-oriented programming."
    )
    assert assistant.last_assessment.status == "SUFFICIENT"
    assert assistant.last_selection.top_relevance == 5.0
    # NLI only ever sees clean evidence sentences, never document headers.
    assert nli.premises
    assert all("[Document" not in premise for premise in nli.premises)


def test_assistant_refuses_when_nothing_is_verified():
    assistant, _ = build_assistant(entails=False)

    response = assistant.chat("What is Python?")

    assert response == Assistant.UNSUPPORTED_ANSWER_MESSAGE
    assert assistant.last_assessment.status == "INSUFFICIENT"
    assert assistant.last_assessment.metrics["rejected_count"] > 0


def test_assistant_refuses_when_best_sentence_is_not_relevant_enough():
    assistant, _ = build_assistant(
        rerank_scores={
            "What is Python?": {
                "Python is a programming language.": -6.0,
                "Python supports object-oriented programming.": -7.0,
            }
        },
        answerability_checker=AnswerabilityChecker(min_top_relevance=0.0),
    )

    response = assistant.chat("What is Python?")

    assert response == Assistant.UNSUPPORTED_ANSWER_MESSAGE

    # A6 extractor rejects candidates below its absolute relevance threshold.
    # Therefore selection and answerability assessment are not reached.
    assert assistant.last_selection is None
    assert assistant.last_assessment is None


def test_assistant_resets_diagnostics_between_calls():
    assistant, _ = build_assistant()

    assistant.chat("What is Python?")
    assert assistant.last_assessment is not None

    assistant.context_validator.is_relevant.return_value = False

    assert (
        assistant.chat("What is Java?")
        == Assistant.NO_KNOWLEDGE_MESSAGE
    )
    assert assistant.last_assessment is None
    assert assistant.last_selection is None