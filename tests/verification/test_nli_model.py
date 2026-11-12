from types import SimpleNamespace

import pytest
import torch

from cognexa.verification import nli_model
from cognexa.verification.nli_model import NLIModel


class FakeTokenizer:
    def __call__(
        self,
        premises,
        hypotheses,
        **kwargs,
    ):
        return {
            "input_ids": torch.ones(
                (len(premises), 8),
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (len(premises), 8),
                dtype=torch.long,
            ),
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            id2label={
                0: "contradiction",
                1: "neutral",
                2: "entailment",
            }
        )

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]

        logits = torch.tensor(
            [[0.1, 0.2, 2.0]] * batch_size
        )

        return SimpleNamespace(logits=logits)


@pytest.fixture
def fake_nli(monkeypatch):
    monkeypatch.setattr(
        nli_model.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    monkeypatch.setattr(
        nli_model.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    return NLIModel(
        model_name="fake-model",
        device="cpu",
    )


def test_nli_model_initialization(fake_nli):
    assert fake_nli.tokenizer is not None
    assert fake_nli.model is not None
    assert fake_nli.device == "cpu"


def test_nli_model_label_mapping(fake_nli):
    assert fake_nli.label_indices == {
        "contradiction": 0,
        "neutral": 1,
        "entailment": 2,
    }


def test_predict_returns_nli_scores(fake_nli):
    result = fake_nli.predict(
        premise="Python is a programming language.",
        hypothesis="Python is a programming language.",
    )

    assert 0.0 <= result.entailment <= 1.0
    assert 0.0 <= result.contradiction <= 1.0
    assert 0.0 <= result.neutral <= 1.0

    assert result.entailment > result.contradiction


def test_predict_batch_returns_correct_number_of_results(fake_nli):
    results = fake_nli.predict_batch(
        premises=[
            "Python is a programming language.",
            "Java is a programming language.",
        ],
        hypotheses=[
            "Python is a programming language.",
            "Java is a programming language.",
        ],
    )

    assert len(results) == 2


def test_predict_batch_empty_input(fake_nli):
    assert fake_nli.predict_batch([], []) == []


def test_predict_batch_rejects_different_lengths(fake_nli):
    with pytest.raises(ValueError):
        fake_nli.predict_batch(
            premises=["Python is popular."],
            hypotheses=[],
        )


def test_predict_batch_rejects_invalid_input(fake_nli):
    with pytest.raises(TypeError):
        fake_nli.predict_batch(
            premises=["Python is popular."],
            hypotheses=[123],
        )


def test_nli_model_rejects_invalid_max_length():
    with pytest.raises(ValueError):
        NLIModel(max_length=0)
