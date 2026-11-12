from collections.abc import Mapping

from cognexa.verification.preprocessing import TextPreprocessor


def _normalize(text: str) -> str:
    return TextPreprocessor.normalize(text)


def _document_content(document) -> str:
    if isinstance(document, str):
        return document

    if isinstance(document, Mapping):
        return (
            document.get("content")
            or document.get("page_content")
            or ""
        )

    return (
        getattr(document, "page_content", "")
        or getattr(document, "content", "")
        or ""
    )


def calculate_evidence_recall(
    contexts,
    expected_evidence,
) -> tuple[int, int, float]:
    """
    Measure how many expected evidence phrases occur in the supplied contexts.

    This is deterministic lexical evidence coverage, not semantic entailment.
    Multiple contexts are evaluated collectively.
    """
    valid_evidence = [
        _normalize(item)
        for item in (expected_evidence or [])
        if isinstance(item, str) and _normalize(item)
    ]

    normalized_contexts = [
        _normalize(context)
        for context in (contexts or [])
        if isinstance(context, str) and _normalize(context)
    ]

    total = len(valid_evidence)

    matched = sum(
        1
        for evidence in valid_evidence
        if any(evidence in context for context in normalized_contexts)
    )

    recall = matched / total if total else 0.0

    return matched, total, recall


def calculate_retrieved_evidence_recall(
    documents,
    expected_evidence,
) -> tuple[int, int, float]:
    contexts = [
        _document_content(document)
        for document in (documents or [])
    ]

    return calculate_evidence_recall(
        contexts=contexts,
        expected_evidence=expected_evidence,
    )


def calculate_context_evidence_recall(
    context: str,
    expected_evidence,
) -> tuple[int, int, float]:
    return calculate_evidence_recall(
        contexts=[context or ""],
        expected_evidence=expected_evidence,
    )
