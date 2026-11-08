
import re

from cognexa.assistant.assistant import Assistant


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def calculate_lexical_answer_point_recall(
    response: str,
    expected_points: list[str],
) -> tuple[int, int, float]:
    """Measure exact lexical coverage of expected answer points."""

    if not expected_points:
        return 0, 0, 0.0

    normalized_response = normalize(response)

    matched_points = sum(
        1
        for point in expected_points
        if normalize(point)
        and normalize(point) in normalized_response
    )

    total_points = len(expected_points)

    return (
        matched_points,
        total_points,
        matched_points / total_points,
    )


def calculate_semantic_answer_point_recall(
    response: str,
    expected_points: list[str],
    answer_verifier,
) -> tuple[int, int, float]:
    """
    Check whether the response semantically supports
    each expected answer point.

    This is an NLI-based approximation, not a human evaluation.
    """

    if not expected_points or not response.strip():
        return 0, len(expected_points), 0.0

    matched_points = 0

    for point in expected_points:
        result = answer_verifier.verify_sentence(
            context=response,
            hypothesis=point,
        )

        if result["status"] == answer_verifier.SUPPORTED:
            matched_points += 1

    total_points = len(expected_points)

    return (
        matched_points,
        total_points,
        matched_points / total_points,
    )


def calculate_evidence_support_rate(
    response: str,
    context: str,
    answer_verifier,
) -> tuple[int, int, float]:
    """Measure how many response sentences are supported by context."""

    sentences = split_sentences(response)

    if not sentences or not context:
        return 0, len(sentences), 0.0

    supported_count = 0

    for sentence in sentences:
        result = answer_verifier.verify_sentence(
            context=context,
            hypothesis=sentence,
        )

        if result["status"] == answer_verifier.SUPPORTED:
            supported_count += 1

    return (
        supported_count,
        len(sentences),
        supported_count / len(sentences),
    )


def is_rejection_response(response: str) -> bool:
    """Check whether the Assistant returns an explicit rejection."""

    normalized_response = normalize(response)

    rejection_messages = [
        Assistant.NO_KNOWLEDGE_MESSAGE,
        Assistant.UNSUPPORTED_ANSWER_MESSAGE,
    ]

    return normalized_response in {
        normalize(message)
        for message in rejection_messages
    }


def test_end_to_end_answer_quality(
    rag_environment,
    evaluation_dataset,
):
    """
    Evaluate the complete RAG answer pipeline.

    Metrics:
    - Non-rejection rate on answerable questions
    - Lexical answer point recall
    - Semantic answer point recall
    - Evidence support rate
    - Unanswerable rejection rate
    """

    assistant = Assistant(
        generator=None,
        prompt_builder=None,
        knowledge_base=rag_environment.knowledge_base,
        context_validator=rag_environment.context_validator,
        answer_verifier=rag_environment.answer_verifier,
        answer_extractor=rag_environment.answer_extractor,
        retrieval_top_k=5,
        query_decomposer=rag_environment.decomposer,
        result_aggregator=rag_environment.result_aggregator,
        retrieval_reranker=rag_environment.reranker,
        reranking_top_k=5,
    )

    questions = evaluation_dataset["questions"]

    answerable_questions = [
        item for item in questions if item["answerable"]
    ]

    unanswerable_questions = [
        item for item in questions if not item["answerable"]
    ]

    answered_count = 0
    rejected_count = 0

    total_expected_points = 0
    lexical_matched_points = 0
    semantic_matched_points = 0

    total_supported_sentences = 0
    total_answer_sentences = 0

    question_results = []

    print("\n" + "=" * 80)
    print("END-TO-END RAG ANSWER QUALITY EVALUATION")
    print("=" * 80)

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        answerable = item["answerable"]

        print("\n" + "-" * 80)
        print(f"QUESTION: {question_id}")
        print(f"TYPE: {item['type']}")
        print(f"QUESTION: {question}")
        print(f"ANSWERABLE: {answerable}")

        response = assistant.chat(question)

        print("\n[ASSISTANT RESPONSE]")
        print(response)

        rejected = is_rejection_response(response)

        if answerable:
            expected_points = item.get(
                "expected_answer_points",
                [],
            )

            total_expected_points += len(expected_points)

            lexical_matched, total, lexical_recall = (
                calculate_lexical_answer_point_recall(
                    response=response,
                    expected_points=expected_points,
                )
            )

            semantic_matched, _, semantic_recall = (
                calculate_semantic_answer_point_recall(
                    response=response,
                    expected_points=expected_points,
                    answer_verifier=assistant.answer_verifier,
                )
            )

            lexical_matched_points += lexical_matched
            semantic_matched_points += semantic_matched

            if not rejected:
                answered_count += 1

            # Retrieve context again for evidence verification.
            # This is a temporary evaluation approach; ideally,
            # Assistant should expose the context used during chat.
            retrieved_documents = assistant._retrieve_documents(
                question
            )

            if retrieved_documents:
                is_relevant = assistant.context_validator.is_relevant(
                    retrieved_documents
                )
            else:
                is_relevant = False

            if is_relevant:
                expanded_documents = (
                    assistant.knowledge_base.expand_to_parents(
                        retrieved_documents
                    )
                )

                context = assistant._format_context(
                    expanded_documents
                )
            else:
                context = None

            supported, sentence_count, support_rate = (
                calculate_evidence_support_rate(
                    response=response,
                    context=context or "",
                    answer_verifier=assistant.answer_verifier,
                )
            )

            total_supported_sentences += supported
            total_answer_sentences += sentence_count

            result = {
                "id": question_id,
                "answerable": True,
                "rejected": rejected,
                "lexical_point_recall": lexical_recall,
                "semantic_point_recall": semantic_recall,
                "supported_sentences": supported,
                "total_sentences": sentence_count,
                "evidence_support_rate": support_rate,
            }

            print("\n[ANSWER QUALITY]")
            print(
                f"Lexical answer points: "
                f"{lexical_matched}/{total}"
            )
            print(f"Lexical point recall: {lexical_recall:.4f}")

            print(
                f"Semantic answer points: "
                f"{semantic_matched}/{total}"
            )
            print(f"Semantic point recall: {semantic_recall:.4f}")

            print(
                f"Evidence support: "
                f"{supported}/{sentence_count} "
                f"({support_rate:.4f})"
            )

        else:
            if rejected:
                rejected_count += 1

            result = {
                "id": question_id,
                "answerable": False,
                "rejected": rejected,
            }

            print("\n[UNANSWERABLE QUESTION]")
            print(f"Correctly rejected: {rejected}")

        question_results.append(result)

    # Aggregate metrics

    non_rejection_rate = (
        answered_count / len(answerable_questions)
        if answerable_questions
        else 0.0
    )

    lexical_point_recall = (
        lexical_matched_points / total_expected_points
        if total_expected_points
        else 0.0
    )

    semantic_point_recall = (
        semantic_matched_points / total_expected_points
        if total_expected_points
        else 0.0
    )

    evidence_support_rate = (
        total_supported_sentences / total_answer_sentences
        if total_answer_sentences
        else 0.0
    )

    rejection_rate = (
        rejected_count / len(unanswerable_questions)
        if unanswerable_questions
        else 0.0
    )

    # Final report

    print("\n" + "=" * 80)
    print("FINAL EVALUATION REPORT")
    print("=" * 80)

    print(f"Total questions: {len(questions)}")
    print(f"Answerable questions: {len(answerable_questions)}")
    print(f"Unanswerable questions: {len(unanswerable_questions)}")

    print("\n[METRICS]")

    print(
        f"Non-rejection Rate: "
        f"{non_rejection_rate:.4f} "
        f"({answered_count}/{len(answerable_questions)})"
    )

    print(
        f"Lexical Answer Point Recall: "
        f"{lexical_point_recall:.4f} "
        f"({lexical_matched_points}/{total_expected_points})"
    )

    print(
        f"Semantic Answer Point Recall: "
        f"{semantic_point_recall:.4f} "
        f"({semantic_matched_points}/{total_expected_points})"
    )

    print(
        f"Evidence Support Rate: "
        f"{evidence_support_rate:.4f} "
        f"({total_supported_sentences}/{total_answer_sentences})"
    )

    print(
        f"Unanswerable Rejection Rate: "
        f"{rejection_rate:.4f} "
        f"({rejected_count}/{len(unanswerable_questions)})"
    )

    print("\n[QUESTION-LEVEL RESULTS]")

    for result in question_results:
        print(result)

    print("\n" + "=" * 80)

    # Execution assertions only; no arbitrary quality thresholds.
    assert len(question_results) == len(questions)

    assert all(
        isinstance(result, dict)
        for result in question_results
    )
