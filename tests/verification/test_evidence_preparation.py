import pytest

import cognexa.verification.verifier as verifier_module
from cognexa.verification import (
    AnswerVerifier,
    EvidenceMatcher,
    NLIScores,
    SentenceStatus,
    TextPreprocessor,
)


class RecordingNLIModel:
    """Fake NLI model that records premises and always entails."""

    def __init__(self):
        self.premises = []
        self.hypotheses = []

    def predict_batch(self, premises, hypotheses):
        self.premises.extend(premises)
        self.hypotheses.extend(hypotheses)

        return [NLIScores(0.95, 0.02, 0.03) for _ in premises]


CONTEXT = (
    "[Document 1 | Source: overfitting.txt | Parent: 0]\n"
    "Overfitting means a model learns noise. It hurts generalization.\n"
    "Regularization reduces overfitting\n"
    "\n"
    "[Document 2 | Source: overfitting.txt | Type: parent]\n"
    "Early stopping also reduces overfitting."
)


# =========================
# TextPreprocessor.split_evidence
# =========================

def test_split_evidence_drops_headers_and_splits_lines():
    assert TextPreprocessor.split_evidence(CONTEXT) == [
        "Overfitting means a model learns noise.",
        "It hurts generalization.",
        "Regularization reduces overfitting",
        "Early stopping also reduces overfitting.",
    ]


def test_split_evidence_returns_empty_for_header_only_text():
    assert TextPreprocessor.split_evidence(
        "[Document 1 | Source: a.txt]"
    ) == []


def test_split_evidence_rejects_non_string():
    with pytest.raises(TypeError):
        TextPreprocessor.split_evidence(123)


def test_split_sentences_is_unchanged():
    # split_sentences keeps its old contract; only split_evidence is new.
    assert TextPreprocessor.split_sentences(
        "One. Two!\nThree without period\nFour."
    ) == ["One.", "Two!", "Three without period\nFour."]


# =========================
# AnswerVerifier uses clean evidence units
# =========================

def test_verifier_exact_matches_first_sentence_of_a_document():
    model = RecordingNLIModel()
    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="Overfitting means a model learns noise.",
        evidence=[CONTEXT],
    )

    assert result.sentences[0].status == SentenceStatus.SUPPORTED
    # The premise is exactly the evidence sentence, without the header.
    assert model.premises == ["Overfitting means a model learns noise."]


def test_verifier_exact_matches_line_without_final_punctuation():
    model = RecordingNLIModel()
    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="Regularization reduces overfitting",
        evidence=[CONTEXT],
    )

    assert result.is_supported
    assert model.premises == ["Regularization reduces overfitting"]


def test_verifier_never_sends_document_headers_to_nli():
    model = RecordingNLIModel()
    verifier = AnswerVerifier(nli_model=model)

    verifier.verify(
        answer="Early stopping can reduce overfitting in models.",
        evidence=[CONTEXT],
    )

    assert model.premises
    assert all("[Document" not in premise for premise in model.premises)


# =========================
# Candidate ordering
# =========================

def test_find_candidate_evidence_orders_by_overlap():
    matcher = EvidenceMatcher()

    result = matcher.find_candidate_evidence(
        "Regularization reduces overfitting in neural networks",
        [
            "Overfitting is common.",
            "Regularization is a technique.",
            "Regularization reduces overfitting in networks.",
        ],
        min_overlap=0.1,
    )

    assert result[0] == "Regularization reduces overfitting in networks."
    # Equal overlap keeps document order.
    assert result[1:] == [
        "Overfitting is common.",
        "Regularization is a technique.",
    ] or result[1:] == [
        "Regularization is a technique.",
        "Overfitting is common.",
    ]


def test_candidate_limit_keeps_best_overlap_not_earliest_sentences():
    model = RecordingNLIModel()
    verifier = AnswerVerifier(
        nli_model=model,
        candidate_limit=2,
        min_overlap=0.2,
    )

    weak = [
        f"Overfitting is discussed in chapter {number}."
        for number in range(5)
    ]
    best = "Dropout reduces overfitting in deep neural networks."

    verifier.verify(
        answer="Dropout reduces overfitting in neural networks.",
        evidence=[" ".join(weak + [best])],
    )

    assert best in model.premises


# =========================
# Lazy NLI model
# =========================

def test_verifier_does_not_load_nli_model_on_construction(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("NLIModel must not be loaded eagerly.")

    monkeypatch.setattr(verifier_module, "NLIModel", fail)

    verifier = AnswerVerifier()

    # No evidence means no NLI call, so the model is still not needed.
    result = verifier.verify(answer="Anything.", evidence=[])

    assert result.unsupported_count == 1


def test_verifier_loads_nli_model_on_first_use(monkeypatch):
    created = []

    def factory():
        model = RecordingNLIModel()
        created.append(model)
        return model

    monkeypatch.setattr(verifier_module, "NLIModel", factory)

    verifier = AnswerVerifier()

    assert created == []

    verifier.verify(
        answer="Python is a language.",
        evidence=["Python is a language."],
    )
    verifier.verify(
        answer="Python is a language.",
        evidence=["Python is a language."],
    )

    assert len(created) == 1
    assert verifier.nli_model is created[0]

# =========================
# Evidence list handling
# =========================

def test_split_evidence_preserves_colon_lead_in_for_each_bullet():
    context = """
    Several techniques help reduce overfitting:
    - L1 regularization
    - L2 regularization
    - Dropout
    - Early stopping
    """

    assert TextPreprocessor.split_evidence(context) == [
        "Several techniques help reduce overfitting: L1 regularization.",
        "Several techniques help reduce overfitting: L2 regularization.",
        "Several techniques help reduce overfitting: Dropout.",
        "Several techniques help reduce overfitting: Early stopping.",
    ]


def test_split_evidence_supports_numbered_lists():
    context = """
    Prevention methods:
    1. Use regularization
    2. Apply dropout
    3. Use early stopping
    """

    assert TextPreprocessor.split_evidence(context) == [
        "Prevention methods: Use regularization.",
        "Prevention methods: Apply dropout.",
        "Prevention methods: Use early stopping.",
    ]


def test_split_evidence_supports_bullet_symbols():
    context = """
    Techniques:
    • Dropout
    * L1 regularization
    - L2 regularization
    """

    assert TextPreprocessor.split_evidence(context) == [
        "Techniques: Dropout.",
        "Techniques: L1 regularization.",
        "Techniques: L2 regularization.",
    ]


def test_split_evidence_does_not_duplicate_consumed_lead_in():
    context = """
    Available methods:
    - Dropout
    - Early stopping
    """

    units = TextPreprocessor.split_evidence(context)

    assert len(units) == 2
    assert all(
        unit.startswith("Available methods:")
        for unit in units
    )


def test_split_evidence_deduplicates_list_items():
    context = """
    Techniques:
    - Dropout
    - Dropout
    - Early stopping
    """

    assert TextPreprocessor.split_evidence(context) == [
        "Techniques: Dropout.",
        "Techniques: Early stopping.",
    ]


def test_split_evidence_preserves_lead_in_without_following_list():
    context = """
    Regularization techniques include:
    They help reduce model complexity.
    """

    assert TextPreprocessor.split_evidence(context) == [
        "Regularization techniques include:",
        "They help reduce model complexity.",
    ]
