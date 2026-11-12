from types import SimpleNamespace

import pytest
import torch

import cognexa.knowledge.validation.answer_verifier as verifier_module
from cognexa.knowledge.validation.answer_verifier import AnswerVerifier


def create_verifier(min_score=0.70):
    return AnswerVerifier(
        min_entailment_score=min_score,
    )


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_init_rejects_negative_threshold():
    with pytest.raises(ValueError):
        AnswerVerifier(min_entailment_score=-0.1)


def test_init_rejects_threshold_greater_than_one():
    with pytest.raises(ValueError):
        AnswerVerifier(min_entailment_score=1.1)


def test_init_accepts_boundary_thresholds():
    assert create_verifier(0.0).min_entailment_score == 0.0
    assert create_verifier(1.0).min_entailment_score == 1.0


# ---------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------

def test_split_sentences():
    answer = (
        "Python is a programming language. "
        "It supports object-oriented programming! "
        "It is widely used in AI?"
    )

    result = AnswerVerifier._split_sentences(answer)

    assert result == [
        "Python is a programming language.",
        "It supports object-oriented programming!",
        "It is widely used in AI?",
    ]


def test_split_sentences_handles_empty_text():
    assert AnswerVerifier._split_sentences("") == []


def test_split_sentences_handles_whitespace():
    assert AnswerVerifier._split_sentences("   ") == []


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def test_load_model_initializes_tokenizer_and_model(monkeypatch):
    calls = {
        "tokenizer": 0,
        "model": 0,
    }

    class FakeTokenizer:
        pass

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(
                id2label={
                    0: "CONTRADICTION",
                    1: "ENTAILMENT",
                    2: "NEUTRAL",
                }
            )

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_tokenizer_loader(model_name):
        calls["tokenizer"] += 1
        return FakeTokenizer()

    def fake_model_loader(model_name):
        calls["model"] += 1
        return FakeModel()

    monkeypatch.setattr(
        verifier_module,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=fake_tokenizer_loader,
        ),
    )

    monkeypatch.setattr(
        verifier_module,
        "AutoModelForSequenceClassification",
        SimpleNamespace(
            from_pretrained=fake_model_loader,
        ),
    )

    verifier = create_verifier()

    verifier._load_model()

    assert verifier.tokenizer is not None
    assert verifier.model is not None
    assert verifier.entailment_label_id == 1

    assert calls == {
        "tokenizer": 1,
        "model": 1,
    }


def test_load_model_does_not_reload_existing_model(monkeypatch):
    calls = {
        "tokenizer": 0,
        "model": 0,
    }

    class FakeModel:
        config = SimpleNamespace(
            id2label={
                0: "CONTRADICTION",
                1: "ENTAILMENT",
            }
        )

        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_tokenizer_loader(model_name):
        calls["tokenizer"] += 1
        return object()

    def fake_model_loader(model_name):
        calls["model"] += 1
        return FakeModel()

    monkeypatch.setattr(
        verifier_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=fake_tokenizer_loader),
    )

    monkeypatch.setattr(
        verifier_module,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=fake_model_loader),
    )

    verifier = create_verifier()

    verifier._load_model()
    verifier._load_model()

    assert calls == {
        "tokenizer": 1,
        "model": 1,
    }


def test_load_model_rejects_missing_entailment_label(monkeypatch):
    class FakeModel:
        config = SimpleNamespace(
            id2label={
                0: "CONTRADICTION",
                1: "NEUTRAL",
            }
        )

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        verifier_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda name: object()),
    )

    monkeypatch.setattr(
        verifier_module,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda name: FakeModel()),
    )

    verifier = create_verifier()

    with pytest.raises(
        ValueError,
        match="does not contain an entailment label",
    ):
        verifier._load_model()


# ---------------------------------------------------------
# Entailment score
# ---------------------------------------------------------

def test_get_entailment_score_uses_model_and_softmax(monkeypatch):
    class FakeTokenizer:
        def __call__(
            self,
            context,
            hypothesis,
            **kwargs,
        ):
            assert context == "Python is a programming language."
            assert hypothesis == "Python supports OOP."
            assert kwargs["return_tensors"] == "pt"
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 512

            return {
                "input_ids": torch.tensor([[1, 2, 3]])
            }

    class FakeModel:
        def __call__(self, **inputs):
            assert "input_ids" in inputs

            return SimpleNamespace(
                logits=torch.tensor([[0.0, 2.0, 0.0]])
            )

    verifier = create_verifier()

    verifier.tokenizer = FakeTokenizer()
    verifier.model = FakeModel()
    verifier.entailment_label_id = 1

    score = verifier._get_entailment_score(
        context="Python is a programming language.",
        hypothesis="Python supports OOP.",
    )

    expected = torch.softmax(
        torch.tensor([[0.0, 2.0, 0.0]]),
        dim=-1,
    )[0, 1].item()

    assert score == pytest.approx(expected)


# ---------------------------------------------------------
# Supported sentences
# ---------------------------------------------------------

def test_get_supported_sentences_returns_supported_only(monkeypatch):
    verifier = create_verifier()

    def mock_scores(context, hypothesis):
        if "programming language" in hypothesis:
            return {
                "entailment": 0.95,
                "neutral": 0.04,
                "contradiction": 0.01,
            }

        return {
            "entailment": 0.20,
            "neutral": 0.70,
            "contradiction": 0.10,
        }

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        mock_scores,
    )

    context = "Python is a high-level programming language."

    answer = (
        "Python is a programming language. "
        "Python was created by Guido van Rossum."
    )

    result = verifier.get_supported_sentences(
        context=context,
        answer=answer,
    )

    assert result == [
        "Python is a programming language.",
    ]


def test_get_supported_sentences_accepts_score_equal_to_threshold(
    monkeypatch,
):
    verifier = create_verifier(min_score=0.70)

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.70,
            "neutral": 0.25,
            "contradiction": 0.05,
        },
    )

    result = verifier.get_supported_sentences(
        context="Python provides support for object-oriented development.",
        answer="Python supports OOP.",
    )

    assert result == ["Python supports OOP."]


def test_get_supported_sentences_rejects_score_below_threshold(
    monkeypatch,
):
    verifier = create_verifier(min_score=0.70)

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.69,
            "neutral": 0.30,
            "contradiction": 0.01,
        },
    )

    result = verifier.get_supported_sentences(
        context="Python provides support for object-oriented development.",
        answer="Python supports OOP.",
    )

    assert result == []


def test_get_supported_sentences_returns_empty_for_empty_context():
    verifier = create_verifier()

    assert verifier.get_supported_sentences(
        context="",
        answer="Python supports OOP.",
    ) == []


def test_get_supported_sentences_returns_empty_for_empty_answer():
    verifier = create_verifier()

    assert verifier.get_supported_sentences(
        context="Python supports OOP.",
        answer="",
    ) == []


# ---------------------------------------------------------
# is_supported
# ---------------------------------------------------------

def test_all_sentences_supported(monkeypatch):
    verifier = create_verifier()

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.95,
            "neutral": 0.04,
            "contradiction": 0.01,
        },
    )

    context = (
        "Python is a high-level programming language. "
        "Python supports object-oriented programming."
    )

    answer = (
        "Python is a high-level programming language. "
        "Python supports object-oriented programming."
    )

    assert verifier.is_supported(context, answer) is True


def test_one_sentence_not_supported(monkeypatch):
    verifier = create_verifier()

    def mock_scores(context, hypothesis):
        if "Guido" in hypothesis:
            return {
                "entailment": 0.20,
                "neutral": 0.70,
                "contradiction": 0.10,
            }

        return {
            "entailment": 0.95,
            "neutral": 0.04,
            "contradiction": 0.01,
        }

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        mock_scores,
    )

    context = "Python is a high-level programming language."

    answer = (
        "Python is a high-level programming language. "
        "Python was created by Guido van Rossum."
    )

    assert verifier.is_supported(context, answer) is False


def test_empty_context():
    verifier = create_verifier()

    assert verifier.is_supported(
        context="",
        answer="Python is a programming language.",
    ) is False


def test_empty_answer():
    verifier = create_verifier()

    assert verifier.is_supported(
        context="Python is a programming language.",
        answer="",
    ) is False


def test_score_below_threshold(monkeypatch):
    verifier = create_verifier()

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.60,
            "neutral": 0.35,
            "contradiction": 0.05,
        },
    )

    assert verifier.is_supported(
        context="Python provides support for object-oriented development.",
        answer="Python supports OOP.",
    ) is False


def test_is_supported_returns_false_when_answer_has_no_sentences():
    verifier = create_verifier()

    assert verifier.is_supported(
        context="Python supports OOP.",
        answer="   ",
    ) is False


def test_is_supported_returns_false_when_no_sentences_are_supported(
    monkeypatch,
):
    verifier = create_verifier()

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.20,
            "neutral": 0.70,
            "contradiction": 0.10,
        },
    )

    result = verifier.is_supported(
        context="Python is a programming language.",
        answer="Python was created in 1995.",
    )

    assert result is False


def test_is_supported_returns_false_when_sentence_split_is_empty(
    monkeypatch,
):
    verifier = create_verifier()

    monkeypatch.setattr(
        verifier,
        "_split_sentences",
        lambda answer: [],
    )

    result = verifier.is_supported(
        context="Python is a programming language.",
        answer="Some non-empty answer",
    )

    assert result is False

def test_verify_sentence_returns_uncertain(verifier, monkeypatch):
    monkeypatch.setattr(
        verifier,
        "_find_exact_evidence",
        lambda context, hypothesis: None,
    )

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.60,
            "neutral": 0.30,
            "contradiction": 0.10,
        },
    )

    result = verifier.verify_sentence(
        context="Regularization reduces overfitting.",
        hypothesis="Regularization helps control model complexity.",
    )

    assert result["status"] == AnswerVerifier.UNCERTAIN
    assert result["method"] == "nli"

def test_verify_sentence_returns_uncertain(monkeypatch):
    verifier = AnswerVerifier(
        min_entailment_score=0.70,
        uncertain_margin=0.15,
    )

    monkeypatch.setattr(
        verifier,
        "_find_exact_evidence",
        lambda context, hypothesis: None,
    )

    monkeypatch.setattr(
        verifier,
        "_get_nli_scores",
        lambda context, hypothesis: {
            "entailment": 0.60,
            "neutral": 0.30,
            "contradiction": 0.10,
        },
    )

    result = verifier.verify_sentence(
        context="Regularization reduces overfitting.",
        hypothesis="Regularization helps control model complexity.",
    )

    assert result["status"] == AnswerVerifier.UNCERTAIN
    assert result["method"] == "nli"

