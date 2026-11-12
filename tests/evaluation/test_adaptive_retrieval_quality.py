from collections import Counter


QUERY_VARIANTS = {
    "OVF-001": [
        "How does overfitting affect a machine learning model?",
        "What happens when a model memorizes its training data?",
        "Why does overfitting reduce generalization?",
    ],
    "OVF-002": [
        "How does model complexity contribute to overfitting?",
        "Why can a complex model memorize training examples?",
        "Why can too many parameters cause overfitting?",
    ],
    "OVF-003": [
        "What happens when a model trains for too long?",
        "Why can excessive training reduce model generalization?",
        "How does training for too many epochs cause overfitting?",
    ],
    "OVF-004": [
        "Which conditions can cause a machine learning model to overfit?",
        "What factors increase the risk of overfitting?",
        "What are other factors that contribute to overfitting?",
    ],
    "OVF-005": [
        "How does overfitting affect model performance on unseen data?",
        "What problems occur when a model overfits its training data?",
        "What are the consequences of overfitting?",
    ],
    "OVF-006": [
        "What techniques help prevent overfitting in machine learning?",
        "How can we improve a model's ability to generalize?",
        "How can overfitting be reduced?",
    ],
    "OVF-007": [
        "How do overfitting and underfitting differ?",
        "How can we distinguish an overfitted model from an underfitted model?",
        "What is the difference between overfitting and underfitting?",
    ],
    "OVF-008": [
        "How are overfitting and underfitting related?",
        "What do overfitting and underfitting indicate about model complexity?",
        "What is the relationship between overfitting and underfitting?",
    ],
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def calculate_evidence_recall(evidence, expected_evidence):
    if not expected_evidence:
        return 0.0

    contents = [
        normalize(item.chunk.content)
        for item in evidence
    ]

    found = sum(
        any(
            normalize(phrase) in content
            for content in contents
        )
        for phrase in expected_evidence
    )

    return found / len(expected_evidence)


def average(values):
    return sum(values) / len(values) if values else 0.0


def test_adaptive_retrieval_quality(
    adaptive_rag_environment,
    evaluation_dataset,
):
    questions = evaluation_dataset["questions"]

    assert len(questions) == 10

    metrics = {
        "evidence_recall": [],
        "coverage": [],
        "iterations": [],
        "sufficiency_correct": [],
    }

    stop_reasons = Counter()
    question_results = []

    print("\n========== ADAPTIVE RAG QUALITY EVALUATION ==========")
    print(f"Dataset: {evaluation_dataset['dataset_name']}")
    print(f"Total questions: {len(questions)}")

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        answerable = item["answerable"]

        # ----------------------------------------------------
        # 1. Build query plan
        # ----------------------------------------------------

        variants = QUERY_VARIANTS.get(question_id, [])

        decomposed_questions = [
            {
                "aspect": f"evaluation_{index + 1}",
                "question": variant,
            }
            for index, variant in enumerate(variants)
        ]

        query_plan = (
            adaptive_rag_environment.query_plan_builder.build(
                original_question=question,
                decomposed_questions=decomposed_questions,
            )
        )

        # ----------------------------------------------------
        # 2. Run Adaptive Retrieval
        # ----------------------------------------------------

        result = (
            adaptive_rag_environment.adaptive_retriever.retrieve(
                query_plan=query_plan,
            )
        )

        # ----------------------------------------------------
        # 3. Calculate metrics
        # ----------------------------------------------------

        recall = (
            calculate_evidence_recall(
                result.evidence,
                item["expected_evidence"],
            )
            if answerable
            else None
        )

        coverage = result.coverage.overall_score

        predicted_sufficient = result.decision.sufficient

        sufficiency_correct = (
            predicted_sufficient == answerable
        )

        metrics["coverage"].append(coverage)
        metrics["iterations"].append(result.iterations)
        metrics["sufficiency_correct"].append(
            int(sufficiency_correct)
        )

        if recall is not None:
            metrics["evidence_recall"].append(recall)

        stop_reason = str(result.stop_reason)
        stop_reasons[stop_reason] += 1

        question_results.append({
            "id": question_id,
            "answerable": answerable,
            "evidence_count": len(result.evidence),
            "recall": recall,
            "coverage": coverage,
            "predicted_sufficient": predicted_sufficient,
            "sufficiency_correct": sufficiency_correct,
            "iterations": result.iterations,
            "stop_reason": stop_reason,
        })

        # ----------------------------------------------------
        # 4. Print question-level results
        # ----------------------------------------------------

        print(f"\n[{question_id}] {question}")
        print(f"  Answerable: {answerable}")
        print(f"  Evidence count: {len(result.evidence)}")

        if recall is not None:
            print(f"  Evidence Recall: {recall:.4f}")

        print(f"  Coverage: {coverage:.4f}")
        print(f"  Predicted sufficient: {predicted_sufficient}")
        print(f"  Sufficiency correct: {sufficiency_correct}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Stop reason: {stop_reason}")

    # ========================================================
    # 5. Aggregate results
    # ========================================================

    print("\n========== AGGREGATE ADAPTIVE RAG RESULTS ==========")

    print(
        "Evidence Recall: "
        f"{average(metrics['evidence_recall']):.4f}"
    )

    print(
        "Average Coverage: "
        f"{average(metrics['coverage']):.4f}"
    )

    print(
        "Sufficiency Accuracy: "
        f"{average(metrics['sufficiency_correct']):.4f}"
    )

    print(
        "Average Iterations: "
        f"{average(metrics['iterations']):.4f}"
    )

    print("\n========== STOP REASON DISTRIBUTION ==========")

    for reason, count in stop_reasons.items():
        print(f"  {reason}: {count}/10")

    # ========================================================
    # 6. Answerable vs Out-of-scope
    # ========================================================

    answerable_results = [
        result
        for result in question_results
        if result["answerable"]
    ]

    out_of_scope_results = [
        result
        for result in question_results
        if not result["answerable"]
    ]

    print("\n========== ANSWERABLE QUESTIONS ==========")

    print(
        "Average Coverage: "
        f"{average([r['coverage'] for r in answerable_results]):.4f}"
    )

    print("\n========== OUT-OF-SCOPE QUESTIONS ==========")

    print(
        "Average Coverage: "
        f"{average([r['coverage'] for r in out_of_scope_results]):.4f}"
    )

    correctly_rejected = sum(
        not r["predicted_sufficient"]
        for r in out_of_scope_results
    )

    print(
        "Correctly rejected: "
        f"{correctly_rejected}/{len(out_of_scope_results)}"
    )

    # --------------------------------------------------------
    # 7. Basic structural validation
    # --------------------------------------------------------

    assert len(question_results) == 10
    assert all(
        1 <= result["iterations"] <= 3
        for result in question_results
    )

    assert all(
        0.0 <= result["coverage"] <= 1.0
        for result in question_results
    )
