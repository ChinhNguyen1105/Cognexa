import pytest

from cognexa.knowledge.validation.context_validator import (
    ContextValidator,
)


def test_is_relevant_returns_false_for_empty_documents():
    validator = ContextValidator(max_distance=0.28)

    assert validator.is_relevant([]) is False


def test_is_relevant_returns_true_for_relevant_document():
    validator = ContextValidator(max_distance=0.28)

    documents = [
        {"distance": 0.1658}
    ]

    assert validator.is_relevant(documents) is True


def test_is_relevant_returns_false_for_unrelated_document():
    validator = ContextValidator(max_distance=0.28)

    documents = [
        {"distance": 1.0742}
    ]

    assert validator.is_relevant(documents) is False


def test_is_relevant_returns_true_when_distance_equals_threshold():
    validator = ContextValidator(max_distance=0.28)

    documents = [
        {"distance": 0.28}
    ]

    assert validator.is_relevant(documents) is True


def test_is_relevant_skips_documents_without_distance():
    validator = ContextValidator(max_distance=0.28)

    documents = [
        {},
        {"distance": 0.5},
    ]

    assert validator.is_relevant(documents) is False


def test_is_relevant_returns_true_when_any_document_is_relevant():
    validator = ContextValidator(max_distance=0.28)

    documents = [
        {"distance": 0.5},
        {"distance": 0.2},
        {"distance": 0.8},
    ]

    assert validator.is_relevant(documents) is True


def test_init_rejects_negative_max_distance():
    with pytest.raises(
        ValueError,
        match="max_distance must be greater than or equal to 0",
    ):
        ContextValidator(max_distance=-0.1)


