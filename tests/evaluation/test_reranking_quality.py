import pytest


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def find_relevant_documents(
    results: list[dict],
    expected_evidence: list[str],
) -> list[bool]:
    """
    A chunk is relevant if it contains at least
    one expected evidence phrase.
    """

    evidence = [
        normalize(item)
        for item in expected_evidence
    ]

    relevance = []

    for result in results:
        content = normalize(result["content"])

        relevance.append(
            any(
                phrase in content
                for phrase in evidence
            )
        )

    return relevance


def calculate_evidence_recall_at_k(
    results: list[dict],
    expected_evidence: list[str],
) -> float:
    """
    Calculate the proportion of expected evidence
    phrases found in the retrieved documents.
    """

    if not expected_evidence:
        return 0.0

    contents = [
        normalize(result["content"])
        for result in results
    ]

    found = sum(
        any(
            normalize(phrase) in content
            for content in contents
        )
        for phrase in expected_evidence
    )

    return found / len(expected_evidence)


def calculate_precision_at_k(
    relevance: list[bool],
) -> float:
    """
    Calculate the proportion of relevant chunks
    among the returned chunks.
    """

    if not relevance:
        return 0.0

    return sum(relevance) / len(relevance)


def calculate_mrr_at_k(
    relevance: list[bool],
) -> float:
    """
    Calculate the reciprocal rank of the first
    relevant chunk.
    """

    for rank, is_relevant in enumerate(
        relevance,
        start=1,
    ):
        if is_relevant:
            return 1.0 / rank

    return 0.0


def evaluate_results(
    results: list[dict],
    expected_evidence: list[str],
) -> dict:
    """
    Calculate all retrieval metrics for a result list.
    """

    relevance = find_relevant_documents(
        results=results,
        expected_evidence=expected_evidence,
    )

    return {
        "recall": calculate_evidence_recall_at_k(
            results=results,
            expected_evidence=expected_evidence,
        ),
        "precision": calculate_precision_at_k(
            relevance=relevance,
        ),
        "mrr": calculate_mrr_at_k(
            relevance=relevance,
        ),
        "relevant_chunks": sum(relevance),
        "retrieved_chunks": len(results),
    }


@pytest.mark.parametrize(
    "top_k",
    [3, 5],
)
def test_cross_encoder_reranking_quality(
    rag_environment,
    evaluation_dataset,
    top_k,
):
    """
    Compare Dense Retrieval against Dense Retrieval
    followed by CrossEncoder Reranking.

    Candidate pool:
        Dense Retrieval retrieves up to 5 chunks.

    Comparison:
        1. Dense Top K
        2. Reranked Top K from the same candidate pool
    """

    candidate_pool_k = 5

    questions = evaluation_dataset["questions"]

    answerable_questions = [
        item
        for item in questions
        if item["answerable"]
    ]

    assert answerable_questions, (
        "No answerable questions found in evaluation dataset."
    )

    total_dense_recall = 0.0
    total_dense_precision = 0.0
    total_dense_mrr = 0.0

    total_reranked_recall = 0.0
    total_reranked_precision = 0.0
    total_reranked_mrr = 0.0

    evaluated_count = 0

    for item in answerable_questions:
        question = item["question"]
        expected_evidence = item["expected_evidence"]

        # Step 1: Retrieve the candidate pool using Dense Retrieval.
        candidate_pool = (
            rag_environment.knowledge_base.search(
                query=question,
                top_k=candidate_pool_k,
            )
        )

        # Dense baseline: take the first K candidates.
        dense_results = candidate_pool[:top_k]

        # Step 2: Rerank the entire candidate pool.
        if candidate_pool:
            candidates = [
                document["content"]
                for document in candidate_pool
            ]

            reranked_items = (
                rag_environment.reranker.rerank(
                    query=question,
                    candidates=candidates,
                )
            )

            # Reorder original documents using CrossEncoder indices.
            reranked_results = [
                candidate_pool[item["index"]]
                for item in reranked_items[:top_k]
            ]
        else:
            reranked_results = []

        # Step 3: Evaluate both result lists.
        dense_metrics = evaluate_results(
            results=dense_results,
            expected_evidence=expected_evidence,
        )

        reranked_metrics = evaluate_results(
            results=reranked_results,
            expected_evidence=expected_evidence,
        )

        total_dense_recall += dense_metrics["recall"]
        total_dense_precision += dense_metrics["precision"]
        total_dense_mrr += dense_metrics["mrr"]

        total_reranked_recall += reranked_metrics["recall"]
        total_reranked_precision += reranked_metrics["precision"]
        total_reranked_mrr += reranked_metrics["mrr"]

        evaluated_count += 1

        print(
            f"\n[{item['id']}] {question}"
        )

        print(
            f"\nDense Top {top_k}:"
            f"\n  Recall: {dense_metrics['recall']:.4f}"
            f"\n  Precision: {dense_metrics['precision']:.4f}"
            f"\n  MRR: {dense_metrics['mrr']:.4f}"
            f"\n  Relevant chunks: "
            f"{dense_metrics['relevant_chunks']}"
            f"/{dense_metrics['retrieved_chunks']}"
        )

        print(
            f"\nCrossEncoder Top {top_k}:"
            f"\n  Recall: {reranked_metrics['recall']:.4f}"
            f"\n  Precision: {reranked_metrics['precision']:.4f}"
            f"\n  MRR: {reranked_metrics['mrr']:.4f}"
            f"\n  Relevant chunks: "
            f"{reranked_metrics['relevant_chunks']}"
            f"/{reranked_metrics['retrieved_chunks']}"
        )

        print(
            f"\n  MRR improvement: "
            f"{reranked_metrics['mrr'] - dense_metrics['mrr']:+.4f}"
        )

    count = evaluated_count

    dense_average = {
        "recall": total_dense_recall / count,
        "precision": total_dense_precision / count,
        "mrr": total_dense_mrr / count,
    }

    reranked_average = {
        "recall": total_reranked_recall / count,
        "precision": total_reranked_precision / count,
        "mrr": total_reranked_mrr / count,
    }

    print(
        "\n========== RERANKING EVALUATION =========="
    )

    print(f"Dataset: {evaluation_dataset['dataset_name']}")
    print(f"Questions evaluated: {count}")
    print(f"Candidate pool size: {candidate_pool_k}")
    print(f"Evaluation Top K: {top_k}")

    print("\nDense Retrieval:")
    print(f"  Evidence Recall@{top_k}: {dense_average['recall']:.4f}")
    print(f"  Chunk Precision@{top_k}: {dense_average['precision']:.4f}")
    print(f"  MRR@{top_k}: {dense_average['mrr']:.4f}")

    print("\nDense + CrossEncoder:")
    print(f"  Evidence Recall@{top_k}: {reranked_average['recall']:.4f}")
    print(f"  Chunk Precision@{top_k}: {reranked_average['precision']:.4f}")
    print(f"  MRR@{top_k}: {reranked_average['mrr']:.4f}")

    print("\nImprovement:")
    print(
        f"  Recall: "
        f"{reranked_average['recall'] - dense_average['recall']:+.4f}"
    )
    print(
        f"  Precision: "
        f"{reranked_average['precision'] - dense_average['precision']:+.4f}"
    )
    print(
        f"  MRR: "
        f"{reranked_average['mrr'] - dense_average['mrr']:+.4f}"
    )
