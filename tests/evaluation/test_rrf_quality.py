import pytest


# ============================================================
# QUERY VARIANTS
# ============================================================

QUERY_VARIANTS = {
    "OVF-001": [
        "What is overfitting in machine learning?",
        "How does overfitting affect a machine learning model?",
        "What happens when a model memorizes its training data?",
    ],
    "OVF-002": [
        "Why can a model with too many parameters overfit?",
        "How does model complexity contribute to overfitting?",
        "Why can a complex model memorize training examples?",
    ],
    "OVF-003": [
        "How does training for too many epochs contribute to overfitting?",
        "What happens when a model trains for too long?",
        "Why can excessive training reduce model generalization?",
    ],
    "OVF-004": [
        "What are other factors that contribute to overfitting?",
        "Which conditions can cause a machine learning model to overfit?",
        "What factors increase the risk of overfitting?",
    ],
    "OVF-005": [
        "What are the consequences of overfitting?",
        "How does overfitting affect model performance on unseen data?",
        "What problems occur when a model overfits its training data?",
    ],
    "OVF-006": [
        "How can overfitting be reduced?",
        "What techniques help prevent overfitting in machine learning?",
        "How can we improve a model's ability to generalize?",
    ],
    "OVF-007": [
        "What is the difference between overfitting and underfitting?",
        "How do overfitting and underfitting differ?",
        "How can we distinguish an overfitted model from an underfitted model?",
    ],
    "OVF-008": [
        "What is the relationship between overfitting and underfitting?",
        "How are overfitting and underfitting related?",
        "What do overfitting and underfitting indicate about model complexity?",
    ],
}


# ============================================================
# METRICS
# ============================================================

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def find_relevant_documents(results, expected_evidence):
    evidence = [normalize(item) for item in expected_evidence]

    relevance = []

    for result in results:
        content = normalize(result["content"])

        is_relevant = any(
            phrase in content
            for phrase in evidence
        )

        relevance.append(is_relevant)

    return relevance


def calculate_evidence_recall_at_k(results, expected_evidence):
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


def calculate_precision_at_k(relevance):
    if not relevance:
        return 0.0

    return sum(relevance) / len(relevance)


def calculate_mrr_at_k(relevance):
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / rank

    return 0.0


def evaluate_results(results, expected_evidence):
    relevance = find_relevant_documents(
        results,
        expected_evidence,
    )

    return {
        "recall": calculate_evidence_recall_at_k(
            results,
            expected_evidence,
        ),
        "precision": calculate_precision_at_k(relevance),
        "mrr": calculate_mrr_at_k(relevance),
        "relevant_chunks": sum(relevance),
        "retrieved_chunks": len(results),
    }


# ============================================================
# EVALUATION
# ============================================================

@pytest.mark.parametrize("top_k", [3, 5])
def test_rrf_retrieval_quality(
    rag_environment,
    evaluation_dataset,
    top_k,
):
    candidate_pool_k = 5

    questions = evaluation_dataset["questions"]

    answerable_questions = [
        item
        for item in questions
        if item["answerable"]
    ]

    assert answerable_questions

    dense_metrics = {
        "recall": [],
        "precision": [],
        "mrr": [],
    }

    rrf_metrics = {
        "recall": [],
        "precision": [],
        "mrr": [],
    }

    print("\n========== RRF RETRIEVAL EVALUATION ==========")
    print(f"Dataset: {evaluation_dataset['dataset_name']}")
    print(f"Questions evaluated: {len(answerable_questions)}")
    print(f"Query variants: 3")
    print(f"Retrieval top K per query: {candidate_pool_k}")
    print(f"Evaluation Top K: {top_k}")

    for item in answerable_questions:
        question_id = item["id"]
        question = item["question"]
        expected_evidence = item["expected_evidence"]

        query_variants = QUERY_VARIANTS[question_id]

        # ----------------------------------------------------
        # 1. Dense Retrieval baseline: original question only
        # ----------------------------------------------------

        dense_results = rag_environment.knowledge_base.search(
            query=question,
            top_k=candidate_pool_k,
        )

        dense_top_k = dense_results[:top_k]

        # ----------------------------------------------------
        # 2. Retrieve documents for each query variant
        # ----------------------------------------------------

        retrieval_results = []

        for query in query_variants:
            results = rag_environment.knowledge_base.search(
                query=query,
                top_k=candidate_pool_k,
            )

            retrieval_results.append(results)

        # ----------------------------------------------------
        # 3. RRF aggregation
        # ----------------------------------------------------

        rrf_results = rag_environment.result_aggregator.aggregate(
            retrieval_results
        )

        rrf_top_k = rrf_results[:top_k]

        # ----------------------------------------------------
        # 4. Evaluate
        # ----------------------------------------------------

        dense_result = evaluate_results(
            dense_top_k,
            expected_evidence,
        )

        rrf_result = evaluate_results(
            rrf_top_k,
            expected_evidence,
        )

        for metric in dense_metrics:
            dense_metrics[metric].append(
                dense_result[metric]
            )

            rrf_metrics[metric].append(
                rrf_result[metric]
            )

        # ----------------------------------------------------
        # 5. Print question results
        # ----------------------------------------------------

        print(f"\n[{question_id}] {question}")

        print("\nDense Retrieval:")
        print(f"  Recall: {dense_result['recall']:.4f}")
        print(f"  Precision: {dense_result['precision']:.4f}")
        print(f"  MRR: {dense_result['mrr']:.4f}")
        print(
            f"  Relevant chunks: "
            f"{dense_result['relevant_chunks']}/"
            f"{dense_result['retrieved_chunks']}"
        )

        print("\nRRF:")
        print(f"  Recall: {rrf_result['recall']:.4f}")
        print(f"  Precision: {rrf_result['precision']:.4f}")
        print(f"  MRR: {rrf_result['mrr']:.4f}")
        print(
            f"  Relevant chunks: "
            f"{rrf_result['relevant_chunks']}/"
            f"{rrf_result['retrieved_chunks']}"
        )

        print(
            f"\n  Recall improvement: "
            f"{rrf_result['recall'] - dense_result['recall']:+.4f}"
        )

        print(
            f"  MRR improvement: "
            f"{rrf_result['mrr'] - dense_result['mrr']:+.4f}"
        )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    def average(values):
        if not values:
            return 0.0

        return sum(values) / len(values)

    print("\n========== AGGREGATE RESULTS ==========")

    print("\nDense Retrieval:")
    for metric, values in dense_metrics.items():
        print(f"  {metric}: {average(values):.4f}")

    print("\nRRF:")
    for metric, values in rrf_metrics.items():
        print(f"  {metric}: {average(values):.4f}")

    print("\nImprovement:")

    for metric in dense_metrics:
        improvement = (
            average(rrf_metrics[metric])
            - average(dense_metrics[metric])
        )

        print(f"  {metric}: {improvement:+.4f}")


