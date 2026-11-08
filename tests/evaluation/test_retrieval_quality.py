import pytest


def normalize(text: str) -> str:
    """
    Normalize text for evidence matching.
    """

    return " ".join(text.lower().split())


def find_relevant_documents(
    results: list[dict],
    expected_evidence: list[str],
) -> list[bool]:
    """
    Determine whether each retrieved chunk contains
    at least one expected evidence phrase.
    """

    evidence = [
        normalize(item)
        for item in expected_evidence
    ]

    relevant = []

    for result in results:
        content = normalize(result["content"])

        is_relevant = any(
            phrase in content
            for phrase in evidence
        )

        relevant.append(is_relevant)

    return relevant


def calculate_evidence_recall_at_k(
    results: list[dict],
    expected_evidence: list[str],
) -> float:
    """
    Measure the proportion of expected evidence phrases
    covered by the retrieved chunks.
    """

    if not expected_evidence:
        return 0.0

    retrieved_contents = [
        normalize(result["content"])
        for result in results
    ]

    found_evidence = 0

    for phrase in expected_evidence:
        normalized_phrase = normalize(phrase)

        if any(
            normalized_phrase in content
            for content in retrieved_contents
        ):
            found_evidence += 1

    return found_evidence / len(expected_evidence)


def calculate_precision_at_k(
    retrieved_relevance: list[bool],
) -> float:
    """
    Measure the proportion of retrieved chunks
    containing at least one expected evidence phrase.
    """

    if not retrieved_relevance:
        return 0.0

    return (
        sum(retrieved_relevance)
        / len(retrieved_relevance)
    )


def calculate_mrr_at_k(
    retrieved_relevance: list[bool],
) -> float:
    """
    Calculate reciprocal rank of the first relevant chunk.
    """

    for rank, is_relevant in enumerate(
        retrieved_relevance,
        start=1,
    ):
        if is_relevant:
            return 1.0 / rank

    return 0.0


@pytest.mark.parametrize(
    "top_k",
    [3, 5],
)
def test_dense_retrieval_quality(
    rag_environment,
    evaluation_dataset,
    top_k,
):
    """
    Evaluate Dense Retrieval using:

    - Evidence Recall@K
    - Chunk Precision@K
    - MRR@K
    """

    questions = evaluation_dataset["questions"]

    answerable_questions = [
        item
        for item in questions
        if item["answerable"]
    ]

    assert answerable_questions, (
        "No answerable questions found in evaluation dataset."
    )

    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0

    evaluated_count = 0

    for item in answerable_questions:
        question = item["question"]

        results = rag_environment.knowledge_base.search(
            query=question,
            top_k=top_k,
        )

        expected_evidence = item["expected_evidence"]

        retrieved_relevance = find_relevant_documents(
            results=results,
            expected_evidence=expected_evidence,
        )

        recall = calculate_evidence_recall_at_k(
            results=results,
            expected_evidence=expected_evidence,
        )

        precision = calculate_precision_at_k(
            retrieved_relevance=retrieved_relevance,
        )

        mrr = calculate_mrr_at_k(
            retrieved_relevance=retrieved_relevance,
        )

        total_recall += recall
        total_precision += precision
        total_mrr += mrr

        evaluated_count += 1

        print(
            f"\n[{item['id']}] {question}"
            f"\nRetrieved chunks: {len(results)}"
            f"\nRelevant chunks: {sum(retrieved_relevance)}"
            f"\nEvidence Recall@{top_k}: {recall:.4f}"
            f"\nChunk Precision@{top_k}: {precision:.4f}"
            f"\nMRR@{top_k}: {mrr:.4f}"
        )

    average_recall = total_recall / evaluated_count
    average_precision = total_precision / evaluated_count
    average_mrr = total_mrr / evaluated_count

    print("\n========== DENSE RETRIEVAL EVALUATION ==========")
    print(f"Dataset: {evaluation_dataset['dataset_name']}")
    print(f"Questions evaluated: {evaluated_count}")
    print(f"Evidence Recall@{top_k}: {average_recall:.4f}")
    print(f"Chunk Precision@{top_k}: {average_precision:.4f}")
    print(f"MRR@{top_k}: {average_mrr:.4f}")

