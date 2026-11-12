from types import SimpleNamespace
from cognexa.assistant.answerability_checker import AnswerabilityChecker


def test_legacy_is_answerable_contract():
    assert AnswerabilityChecker.is_answerable(["Python is a language."])
    assert not AnswerabilityChecker.is_answerable(["", None])


def test_assess_sufficient_selection():
    selection = SimpleNamespace(
        sentences=[{"sentence": "A supported fact."}],
        aspects=["definition"],
        uncovered_aspects=[],
        top_relevance=1.0,
    )
    result = AnswerabilityChecker().assess("What is it?", selection)
    assert result.status == "SUFFICIENT"
    assert result.answerable is True


def test_assess_partial_selection():
    selection = SimpleNamespace(
        sentences=[{"sentence": "A supported fact."}],
        aspects=["causes", "effects"],
        uncovered_aspects=["effects"],
        top_relevance=1.0,
    )
    result = AnswerabilityChecker().assess("What causes and effects?", selection)
    assert result.status == "PARTIAL"
    assert result.answerable is True
    assert result.metrics["uncovered_aspects"] == ["effects"]


def test_assess_insufficient_selection():
    result = AnswerabilityChecker().assess("Question", [])
    assert result.status == "INSUFFICIENT"
    assert result.answerable is False


