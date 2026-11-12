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
    evidence = [
        normalize(item)
        for item in expected_evidence
    ]

    relevance = []

    for result in results:
        content = normalize(result["content"])

        relevance.append(
            any(phrase in content for phrase in evidence)
        )

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


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# PIPELINE EVALUATION
# ============================================================

@pytest.mark.parametrize("top_k", [3, 5])
def test_retrieval_pipeline_quality(
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

    methods = {
        "dense": {
            "recall": [],
            "precision": [],
            "mrr": [],
        },
        "rrf": {
            "recall": [],
            "precision": [],
            "mrr": [],
        },
        "rrf_crossencoder": {
            "recall": [],
            "precision": [],
            "mrr": [],
        },
    }

    print("\n========== RETRIEVAL PIPELINE EVALUATION ==========")
    print(f"Dataset: {evaluation_dataset['dataset_name']}")
    print(f"Questions evaluated: {len(answerable_questions)}")
    print("Query variants: 3")
    print(f"Candidate pool size: {candidate_pool_k}")
    print(f"Evaluation Top K: {top_k}")

    for item in answerable_questions:
        question_id = item["id"]
        question = item["question"]
        expected_evidence = item["expected_evidence"]

        query_variants = QUERY_VARIANTS[question_id]

        # ----------------------------------------------------
        # 1. Dense Retrieval baseline
        # ----------------------------------------------------

        dense_candidates = (
            rag_environment.knowledge_base.search(
                query=question,
                top_k=candidate_pool_k,
            )
        )

        dense_results = dense_candidates[:top_k]

        # ----------------------------------------------------
        # 2. Multi-query Dense Retrieval
        # ----------------------------------------------------

        retrieval_results = []

        for query in query_variants:
            results = rag_environment.knowledge_base.search(
                query=query,
                top_k=candidate_pool_k,
            )

            retrieval_results.append(results)

        # ----------------------------------------------------
        # 3. RRF Aggregation
        # ----------------------------------------------------

        rrf_candidates = (
            rag_environment.result_aggregator.aggregate(
                retrieval_results
            )
        )

        rrf_results = rrf_candidates[:top_k]

        # ----------------------------------------------------
        # 4. CrossEncoder reranking on RRF candidates
        # ----------------------------------------------------

        if rrf_candidates:
            candidates = [
                document["content"]
                for document in rrf_candidates
            ]

            reranked_items = rag_environment.reranker.rerank(
                query=question,
                candidates=candidates,
            )

            reranked_candidates = [
                rrf_candidates[result["index"]]
                for result in reranked_items
            ]

        else:
            reranked_candidates = []

        reranked_results = reranked_candidates[:top_k]

        # ----------------------------------------------------
        # 5. Evaluate all methods
        # ----------------------------------------------------

        results_by_method = {
            "dense": evaluate_results(
                dense_results,
                expected_evidence,
            ),
            "rrf": evaluate_results(
                rrf_results,
                expected_evidence,
            ),
            "rrf_crossencoder": evaluate_results(
                reranked_results,
                expected_evidence,
            ),
        }

        for method, metrics in results_by_method.items():
            for metric in methods[method]:
                methods[method][metric].append(
                    metrics[metric]
                )

        # ----------------------------------------------------
        # 6. Print question-level results
        # ----------------------------------------------------

        print(f"\n[{question_id}] {question}")

        for method, metrics in results_by_method.items():
            print(f"\n{method.upper()}:")

            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  MRR: {metrics['mrr']:.4f}")

            print(
                f"  Relevant chunks: "
                f"{metrics['relevant_chunks']}/"
                f"{metrics['retrieved_chunks']}"
            )

        print("\nImprovements:")

        print(
            "  RRF vs Dense MRR: "
            f"{results_by_method['rrf']['mrr'] - results_by_method['dense']['mrr']:+.4f}"
        )

        print(
            "  CrossEncoder vs RRF MRR: "
            f"{results_by_method['rrf_crossencoder']['mrr'] - results_by_method['rrf']['mrr']:+.4f}"
        )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    print("\n========== AGGREGATE RESULTS ==========")

    for method, metrics in methods.items():
        print(f"\n{method.upper()}:")

        for metric, values in metrics.items():
            print(f"  {metric}: {average(values):.4f}")

    print("\n========== PIPELINE IMPROVEMENTS ==========")

    comparisons = [
        ("RRF vs Dense", "rrf", "dense"),
        (
            "RRF + CrossEncoder vs RRF",
            "rrf_crossencoder",
            "rrf",
        ),
        (
            "RRF + CrossEncoder vs Dense",
            "rrf_crossencoder",
            "dense",
        ),
    ]

    for comparison_name, new_method, base_method in comparisons:
        print(f"\n{comparison_name}:")

        for metric in methods[base_method]:
            new_value = average(methods[new_method][metric])
            base_value = average(methods[base_method][metric])

            print(
                f"  {metric}: "
                f"{new_value - base_value:+.4f}"
            )


