from cognexa.verification.verifier import AnswerVerifier
from cognexa.verification.schemas import (
    NLIScores,
    SentenceStatus,
    VerificationStatus,
)


class FakeNLIModel:
    def __init__(self, scores):
        self.scores = scores
        self.called = False

    def predict_batch(self, premises, hypotheses):
        self.called = True
        assert len(premises) == len(hypotheses)
        assert len(premises) == len(self.scores)
        return self.scores


def test_verify_supported_answer():
    model = FakeNLIModel([
        NLIScores(0.95, 0.02, 0.03),
    ])

    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="Python is a programming language.",
        evidence=[
            "Python is a programming language.",
        ],
    )

    assert model.called
    assert result.status == VerificationStatus.SUPPORTED
    assert result.supported_count == 1


def test_verify_unsupported_when_no_evidence():
    model = FakeNLIModel([])
    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="Python is a programming language.",
        evidence=[],
    )

    assert result.status == VerificationStatus.UNSUPPORTED
    assert result.unsupported_count == 1
    assert not model.called


def test_verify_unsupported_when_no_candidate_found():
    model = FakeNLIModel([])
    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="Python is a programming language.",
        evidence=["The weather is sunny today."],
    )

    assert result.status == VerificationStatus.UNSUPPORTED
    assert result.unsupported_count == 1
    assert not model.called


def test_verify_contradicted_answer():
    model = FakeNLIModel([
        NLIScores(0.02, 0.95, 0.03),
    ])

    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="Python is a compiled language.",
        evidence=[
            "Python is generally described as an interpreted language.",
        ],
    )

    assert result.status == VerificationStatus.CONTRADICTED
    assert result.contradicted_count == 1


def test_verify_multiple_sentences():
    model = FakeNLIModel([
        NLIScores(0.95, 0.02, 0.03),
        NLIScores(0.10, 0.05, 0.85),
    ])

    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer=(
            "Python is a programming language. "
            "Python was created in 1995."
        ),
        evidence=[
            "Python is a programming language.",
            "Python was first released in 1991.",
        ],
    )

    assert result.supported_count == 1
    assert result.unsupported_count == 1
    assert result.status == VerificationStatus.PARTIALLY_SUPPORTED


def test_verify_empty_answer():
    model = FakeNLIModel([])
    verifier = AnswerVerifier(nli_model=model)

    result = verifier.verify(
        answer="",
        evidence=["Python is a programming language."],
    )

    assert result.status == VerificationStatus.UNSUPPORTED
    assert result.sentences == []
    assert not model.called


def test_verify_rejects_invalid_answer():
    verifier = AnswerVerifier(nli_model=FakeNLIModel([]))

    try:
        verifier.verify(answer=123, evidence=[])
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_verify_rejects_invalid_evidence():
    verifier = AnswerVerifier(nli_model=FakeNLIModel([]))

    try:
        verifier.verify(
            answer="Python is popular.",
            evidence=["Python is popular.", 123],
        )
        assert False, "Expected TypeError"
    except TypeError:
        pass
