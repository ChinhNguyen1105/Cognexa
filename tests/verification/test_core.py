import pytest

from cognexa.verification.schemas import (
    NLIScores,
    SentenceStatus,
    SentenceVerification,
    VerificationStatus,
    VerificationResult,
)
from cognexa.verification.preprocessing import TextPreprocessor
from cognexa.verification.matching import EvidenceMatcher
from cognexa.verification.decision import VerificationDecision


# =========================
# Schemas
# =========================

def test_nli_scores_accept_valid_values():
    scores = NLIScores(
        entailment=0.8,
        contradiction=0.1,
        neutral=0.1,
    )

    assert scores.entailment == 0.8


def test_nli_scores_reject_invalid_values():
    with pytest.raises(ValueError):
        NLIScores(
            entailment=1.2,
            contradiction=0.0,
            neutral=0.0,
        )


def test_verification_result_is_supported():
    result = VerificationResult(
        status=VerificationStatus.SUPPORTED,
        sentences=[
            SentenceVerification(
                sentence="Python is a programming language.",
                status=SentenceStatus.SUPPORTED,
            )
        ],
    )

    assert result.is_supported is True


def test_empty_verification_result_is_not_supported():
    result = VerificationResult(
        status=VerificationStatus.SUPPORTED,
    )

    assert result.is_supported is False


# =========================
# Preprocessing
# =========================

def test_normalize_text():
    result = TextPreprocessor.normalize(
        "  PYTHON   Is A PROGRAMMING Language!  "
    )

    assert result == "python is a programming language"


def test_split_sentences():
    result = TextPreprocessor.split_sentences(
        "Python is popular. It is easy to learn! It is versatile."
    )

    assert result == [
        "Python is popular.",
        "It is easy to learn!",
        "It is versatile.",
    ]


def test_split_empty_text():
    assert TextPreprocessor.split_sentences("") == []


def test_normalize_rejects_non_string():
    with pytest.raises(TypeError):
        TextPreprocessor.normalize(123)


# =========================
# Evidence Matching
# =========================

def test_exact_match_ignores_case_and_punctuation():
    matcher = EvidenceMatcher()

    assert matcher.exact_match(
        "Python is popular!",
        ["python is popular"],
    ) is True


def test_find_exact_evidence():
    matcher = EvidenceMatcher()

    result = matcher.find_exact_evidence(
        "Python is popular!",
        [
            "Python is popular",
            "Java is popular",
        ],
    )

    assert result == ["Python is popular"]


def test_find_candidate_evidence():
    matcher = EvidenceMatcher()

    result = matcher.find_candidate_evidence(
        "Python is a programming language",
        [
            "Python is a popular programming language",
            "Java is commonly used for backend development",
        ],
        min_overlap=0.3,
    )

    assert "Python is a popular programming language" in result


def test_candidate_matching_rejects_invalid_threshold():
    matcher = EvidenceMatcher()

    with pytest.raises(ValueError):
        matcher.find_candidate_evidence(
            "Python is popular",
            ["Python is popular"],
            min_overlap=1.5,
        )


# =========================
# Decision
# =========================

@pytest.fixture
def decision():
    return VerificationDecision()


def test_decision_supported(decision):
    result = decision.evaluate(
        sentence="Python is a programming language.",
        scores=NLIScores(0.90, 0.02, 0.08),
        evidence=["Python is a programming language."],
    )

    assert result.status == SentenceStatus.SUPPORTED


def test_decision_contradicted(decision):
    result = decision.evaluate(
        sentence="Python is a compiled language.",
        scores=NLIScores(0.05, 0.90, 0.05),
        evidence=["Python is generally described as interpreted."],
    )

    assert result.status == SentenceStatus.CONTRADICTED


def test_decision_uncertain(decision):
    result = decision.evaluate(
        sentence="Python is used in many applications.",
        scores=NLIScores(0.60, 0.10, 0.30),
        evidence=["Python is used in data science."],
    )

    assert result.status == SentenceStatus.UNCERTAIN


def test_decision_unsupported(decision):
    result = decision.evaluate(
        sentence="Python was created in 1995.",
        scores=NLIScores(0.10, 0.10, 0.80),
        evidence=["Python was first released in 1991."],
    )

    assert result.status == SentenceStatus.UNSUPPORTED


def test_aggregate_all_supported(decision):
    sentences = [
        SentenceVerification("A", SentenceStatus.SUPPORTED),
        SentenceVerification("B", SentenceStatus.SUPPORTED),
    ]

    assert decision.aggregate(sentences) == VerificationStatus.SUPPORTED


def test_aggregate_mixed_results(decision):
    sentences = [
        SentenceVerification("A", SentenceStatus.SUPPORTED),
        SentenceVerification("B", SentenceStatus.UNSUPPORTED),
    ]

    assert decision.aggregate(sentences) == VerificationStatus.PARTIALLY_SUPPORTED


def test_aggregate_all_contradicted(decision):
    sentences = [
        SentenceVerification("A", SentenceStatus.CONTRADICTED),
        SentenceVerification("B", SentenceStatus.CONTRADICTED),
    ]

    assert decision.aggregate(sentences) == VerificationStatus.CONTRADICTED


def test_aggregate_empty_results(decision):
    assert decision.aggregate([]) == VerificationStatus.UNSUPPORTED
from cognexa.verification.decision import VerificationDecision
from cognexa.verification.schemas import (
    NLIScores,
    SentenceStatus,
)


def test_different_candidates_with_conflicting_signals_are_not_uncertain():
    decision = VerificationDecision()

    result = decision.evaluate_candidates(
        sentence="Python is a compiled language.",
        candidates=[
            (
                "Python is a compiled language.",
                NLIScores(0.95, 0.02, 0.03),
            ),
            (
                "Python is an interpreted language.",
                NLIScores(0.02, 0.95, 0.03),
            ),
        ],
    )

    assert result.status == SentenceStatus.SUPPORTED
    assert result.evidence == ["Python is a compiled language."]
    assert result.reason == "Evidence entails the answer."


def test_same_candidate_with_entailment_and_contradiction_is_uncertain():
    decision = VerificationDecision()

    result = decision.evaluate_candidates(
        sentence="Python is a compiled language.",
        candidates=[
            (
                "Conflicting statement.",
                NLIScores(0.90, 0.85, 0.05),
            ),
        ],
    )

    assert result.status == SentenceStatus.UNCERTAIN
    assert result.evidence == ["Conflicting statement."]

def test_different_candidates_with_conflicting_signals_are_not_uncertain():
    decision = VerificationDecision()

    result = decision.evaluate_candidates(
        sentence="Python is a compiled language.",
        candidates=[
            (
                "Python is a compiled language.",
                NLIScores(0.95, 0.02, 0.03),
            ),
            (
                "Python is an interpreted language.",
                NLIScores(0.02, 0.95, 0.03),
            ),
        ],
    )

    assert result.status == SentenceStatus.SUPPORTED
    assert result.evidence == [
        "Python is a compiled language."
    ]
    assert result.reason == "Evidence entails the answer."
