from types import SimpleNamespace

from cognexa.knowledge.evaluation.metrics.context_recall import (
    calculate_context_evidence_recall,
    calculate_evidence_recall,
    calculate_retrieved_evidence_recall,
)


def test_evidence_recall_matches_case_and_punctuation():
    matched, total, recall = calculate_evidence_recall(
        contexts=["Overfitting occurs when a model LEARNS THE TRAINING DATA TOO CLOSELY!"],
        expected_evidence=["LEARNS THE TRAINING DATA TOO CLOSELY"],
    )

    assert (matched, total, recall) == (1, 1, 1.0)


def test_evidence_recall_combines_multiple_documents():
    documents = [
        {"page_content": "Overfitting learns the training data too closely."},
        {"page_content": "It can also memorize noise and accidental patterns."},
    ]

    matched, total, recall = calculate_retrieved_evidence_recall(
        documents=documents,
        expected_evidence=[
            "learns the training data too closely",
            "noise and accidental patterns",
        ],
    )

    assert matched == 2
    assert total == 2
    assert recall == 1.0


def test_context_evidence_recall_uses_actual_context():
    matched, total, recall = calculate_context_evidence_recall(
        context="Regularization helps reduce overfitting.",
        expected_evidence=[
            "Regularization helps reduce overfitting",
            "dropout",
        ],
    )

    assert matched == 1
    assert total == 2
    assert recall == 0.5


def test_empty_expected_evidence_returns_zero():
    assert calculate_evidence_recall([], []) == (0, 0, 0.0)


def test_retrieved_evidence_recall_supports_document_objects():
    documents = [
        SimpleNamespace(page_content="A model memorizes training examples.")
    ]

    matched, total, recall = calculate_retrieved_evidence_recall(
        documents=documents,
        expected_evidence=["memorizes training examples"],
    )

    assert (matched, total, recall) == (1, 1, 1.0)


def test_retrieved_evidence_recall_supports_evidence_like_objects():
    documents = [
        SimpleNamespace(
            document=SimpleNamespace(
                chunk=SimpleNamespace(content="Evidence text from nested chunk")
            )
        )
    ]

    matched, total, recall = calculate_retrieved_evidence_recall(
        documents=documents,
        expected_evidence=["Evidence text from nested chunk"],
    )

    assert (matched, total, recall) == (1, 1, 1.0)


def test_retrieved_evidence_recall_handles_missing_or_empty_content():
    documents = [
        {"content": ""},
        SimpleNamespace(document=None),
        SimpleNamespace(),
        None,
    ]

    matched, total, recall = calculate_retrieved_evidence_recall(
        documents=documents,
        expected_evidence=["anything"],
    )

    assert (matched, total, recall) == (0, 1, 0.0)


def test_retrieved_evidence_recall_supports_string_and_dict_content():
    documents = [
        "A model can overfit when it memorizes training examples.",
        {"content": "Regularization reduces overfitting."},
    ]

    matched, total, recall = calculate_retrieved_evidence_recall(
        documents=documents,
        expected_evidence=[
            "memorizes training examples",
            "Regularization reduces overfitting",
        ],
    )

    assert (matched, total, recall) == (2, 2, 1.0)


def test_unmatched_evidence_returns_zero_recall():
    matched, total, recall = calculate_evidence_recall(
        contexts=["Python is a programming language."],
        expected_evidence=["training data", "validation data"],
    )

    assert matched == 0
    assert total == 2
    assert recall == 0.0


