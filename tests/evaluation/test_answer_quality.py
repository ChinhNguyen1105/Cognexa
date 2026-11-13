import re

from cognexa.assistant.assistant import Assistant
from cognexa.verification import VerificationStatus
from cognexa.knowledge.evaluation.metrics.context_recall import (
    calculate_context_evidence_recall,
    calculate_retrieved_evidence_recall,
)
from cognexa.knowledge.evaluation.metrics.answer_relevance import (
    calculate_answer_question_relevance,
)


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def is_rejection_response(response: str) -> bool:
    if not isinstance(response, str):
        return False
    normalized = normalize(response)
    rejection_messages = {
        normalize(Assistant.NO_KNOWLEDGE_MESSAGE),
        normalize(Assistant.UNSUPPORTED_ANSWER_MESSAGE),
    }
    return normalized in rejection_messages


def calculate_lexical_answer_point_recall(
    response: str,
    expected_points: list[str],
) -> tuple[int, int, float]:
    """
    Token-level recall.

    Returns:
        matched_tokens, total_expected_tokens, recall
    """
    if not expected_points:
        return 0, 0, 0.0

    response_tokens = set(
        re.findall(r"\b\w+\b", (response or "").casefold())
    )

    matched_tokens = 0
    total_tokens = 0

    for point in expected_points:
        expected_tokens = set(
            re.findall(r"\b\w+\b", (point or "").casefold())
        )

        if not expected_tokens:
            continue

        total_tokens += len(expected_tokens)
        matched_tokens += len(expected_tokens & response_tokens)

    recall = matched_tokens / total_tokens if total_tokens else 0.0
    return matched_tokens, total_tokens, recall


def verify_against_context(point: str, context: str, answer_verifier, window: int = 8, stride: int = 3):
    sentences = split_sentences(context)
    if not sentences:
        return False

    for i in range(0, len(sentences), stride):
        premise = " ".join(sentences[i:i + window])
        result = answer_verifier.verify(
            answer=point,
            evidence=[premise],
        )
        if getattr(result, "status", None) == VerificationStatus.SUPPORTED:
            return True

    return False


def calculate_nli_context_coverage(
    context: str,
    expected_points: list[str],
    answer_verifier,
) -> tuple[int, int, float]:
    """
    Measures NLI-based coverage of expected answer points
    using sentence-window evidence windows rather than the full context blob.
    """
    if not expected_points or not context or not context.strip():
        return 0, len(expected_points), 0.0

    matched = 0

    for point in expected_points:
        if verify_against_context(point, context, answer_verifier):
            matched += 1

    total = len(expected_points)
    return matched, total, matched / total if total else 0.0


def calculate_semantic_answer_point_recall(response: str, expected_points: list[str], answer_verifier) -> tuple[int, int, float]:
    if not expected_points or not response or not response.strip():
        return 0, len(expected_points), 0.0

    matched = 0
    for point in expected_points:
        result = answer_verifier.verify(
            answer=point,
            evidence=[response],
        )
        if getattr(result, "status", None) == VerificationStatus.SUPPORTED:
            matched += 1

    total = len(expected_points)
    return matched, total, matched / total if total else 0.0


def calculate_evidence_support_rate(response: str, context: str, answer_verifier) -> tuple[int, int, float]:
    sentences = split_sentences(response)
    if not sentences or not context:
        return 0, len(sentences), 0.0

    supported = 0
    for sentence in sentences:
        if verify_against_context(sentence, context, answer_verifier):
            supported += 1

    total = len(sentences)
    return supported, total, supported / total if total else 0.0



def build_assistant(rag_environment):
    assistant = Assistant(
        generator=rag_environment.generator,
        prompt_builder=rag_environment.prompt_builder,
        knowledge_base=rag_environment.knowledge_base,
        context_validator=rag_environment.context_validator,
        answer_verifier=rag_environment.answer_verifier,
        answer_extractor=rag_environment.answer_extractor,
        answer_selector=rag_environment.answer_selector,
        answerability_checker=getattr(rag_environment, "answerability_checker", None),
        adaptive_retriever=rag_environment.adaptive_retriever,
        query_plan_builder=getattr(rag_environment, "query_plan_builder", None) or __import__("cognexa.knowledge.retrieval.query_plan_builder", fromlist=["QueryPlanBuilder"]).QueryPlanBuilder(),
        evidence_context_builder=getattr(rag_environment, "evidence_context_builder", None) or __import__("cognexa.knowledge.retrieval.evidence_context_builder", fromlist=["EvidenceContextBuilder"]).EvidenceContextBuilder(),
        retrieval_top_k=10,
        query_decomposer=rag_environment.decomposer,
    )

    # A5.1: instrument the exact components used by this benchmark.
    from cognexa.assistant.diagnostics import (
        configure_diagnostics,
        instrument_method,
    )

    logger = configure_diagnostics()
    if not logger.disabled:
        logger.info(
            "A5.1 BENCHMARK PIPELINE: adaptive retriever is attached and legacy KnowledgeBase route is disabled."
        )
        instrument_method(rag_environment.knowledge_base, "search", "KB_SEARCH")
        instrument_method(rag_environment.adaptive_retriever, "retrieve", "ADAPTIVE_RETRIEVER")
        instrument_method(assistant.answer_extractor, "extract_candidates", "EXTRACTOR")
        instrument_method(assistant.answer_verifier, "verify", "VERIFIER")
        instrument_method(assistant.answer_selector, "select_detailed", "SELECTOR")
        instrument_method(assistant.answerability_checker, "assess", "ANSWERABILITY")

    return assistant


def test_end_to_end_answer_quality(rag_environment, evaluation_dataset):
    """Run the end-to-end answer pipeline over the evaluation dataset."""
    assistant = build_assistant(rag_environment)
    from cognexa.assistant.diagnostics import configure_diagnostics
    diagnostic_logger = configure_diagnostics()
    questions = evaluation_dataset["questions"]

    answerable_questions = [item for item in questions if item.get("answerable")]
    unanswerable_questions = [item for item in questions if not item.get("answerable")]

    question_results = []
    answered_count = 0
    rejected_count = 0
    total_expected_points = 0
    total_expected_tokens = 0
    lexical_points_matched = 0
    semantic_points_matched = 0
    context_points_matched = 0
    total_context_points = 0
    retrieved_evidence_matched = 0
    context_evidence_matched = 0
    total_expected_evidence = 0
    total_supported_sentences = 0
    total_answer_sentences = 0
    total_answer_relevance_score = 0.0
    answer_relevance_count = 0

    print("\n" + "=" * 80)
    print("END-TO-END RAG ANSWER QUALITY EVALUATION")
    print("=" * 80)

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        answerable = bool(item.get("answerable"))

        print(f"\n--- {question_id} ---")
        print(f"Question: {question}")
        print(f"Answerable: {answerable}")

        if not diagnostic_logger.disabled:
            diagnostic_logger.info(
                "========== QUESTION START id=%s answerable=%s question=%s ==========",
                question_id,
                answerable,
                question,
            )

        response = assistant.chat(question)
        rejected = is_rejection_response(response)

        if not diagnostic_logger.disabled:
            diagnostic_logger.info(
                "QUESTION END id=%s rejected=%s response=%r",
                question_id,
                rejected,
                response,
            )
            diagnostic_logger.info("========== QUESTION END id=%s ==========", question_id)

        if answerable:
            answer_relevance = calculate_answer_question_relevance(
                question=question,
                answer=response,
                reranker=rag_environment.reranker,
            )

            if answer_relevance is not None:
                total_answer_relevance_score += answer_relevance
                answer_relevance_count += 1

            expected_points = item.get("expected_answer_points") or item.get("expected_evidence") or []
            total_expected_points += len(expected_points)

            lexical_matched, lexical_total, lexical_recall = calculate_lexical_answer_point_recall(
                response=response,
                expected_points=expected_points,
            )
            semantic_matched, semantic_total, semantic_recall = calculate_semantic_answer_point_recall(
                response=response,
                expected_points=expected_points,
                answer_verifier=assistant.answer_verifier,
            )

            lexical_points_matched += lexical_matched
            total_expected_tokens += lexical_total
            semantic_points_matched += semantic_matched

            # Use the exact context consumed by this Assistant.chat() call.
            context = getattr(assistant, "last_context", None) or ""
            expected_evidence = item.get("expected_evidence") or []
            retrieved_documents = getattr(assistant, "last_documents", []) or []

            retrieved_matched, evidence_total, retrieved_recall = (
                calculate_retrieved_evidence_recall(
                    documents=retrieved_documents,
                    expected_evidence=expected_evidence,
                )
            )

            context_evidence_matched_count, context_evidence_total, context_evidence_recall = (
                calculate_context_evidence_recall(
                    context=context,
                    expected_evidence=expected_evidence,
                )
            )

            retrieved_evidence_matched += retrieved_matched
            context_evidence_matched += context_evidence_matched_count
            total_expected_evidence += evidence_total
            if question_id in {"OVF-003", "OVF-006", "OVF-007"}:
                print("\n========== ACTUAL CONTEXT ==========")
                print(context)

                print("\n========== ACTUAL RETRIEVED DOCUMENTS ==========")
                for index, document in enumerate(
                    getattr(assistant, "last_documents", []) or [],
                    start=1,
                ):
                    print(f"\n--- Document {index} ---")
                    print(document)
            context_matched, context_total, context_recall = (
                calculate_nli_context_coverage(
                    context=context,
                    expected_points=expected_points,
                    answer_verifier=assistant.answer_verifier,
                )
            )

            context_points_matched += context_matched
            total_context_points += context_total

            supported, sentence_count, support_rate = calculate_evidence_support_rate(
                response=response,
                context=context,
                answer_verifier=assistant.answer_verifier,
            )

            total_supported_sentences += supported
            total_answer_sentences += sentence_count

            # Expose selector behavior for diagnosis.
            selection = getattr(assistant, "last_selection", None)
            selected_sentences = getattr(selection, "texts", []) or []
            rejected_sentences = getattr(selection, "rejected", []) or []
            uncovered_aspects = getattr(selection, "uncovered_aspects", []) or []
            retrieved_documents = getattr(assistant, "last_documents", []) or []

            print(
                f"Actual retrieval: {len(retrieved_documents)} documents | "
                f"Context characters: {len(context)}"
            )
            print(
                f"Selector: {len(selected_sentences)} selected | "
                f"{len(rejected_sentences)} rejected | "
                f"Uncovered aspects: {uncovered_aspects}"
            )

            for selected_sentence in selected_sentences:
                print(f"  [SELECTED] {selected_sentence}")

            for rejected_sentence in rejected_sentences:
                print(
                    "  [REJECTED] "
                    f"{rejected_sentence.get('sentence')} | "
                    f"status={rejected_sentence.get('status')}"
                )

            if not rejected:
                answered_count += 1

            result = {
                "id": question_id,
                "answerable": True,
                "rejected": rejected,
                "token_recall": lexical_recall,
                "answer_question_relevance": answer_relevance,
                "semantic_point_recall": semantic_recall,
                "nli_context_coverage": context_recall,
                "coverage_gap": context_recall - semantic_recall,
                "retrieved_evidence_recall": retrieved_recall,
                "context_evidence_recall": context_evidence_recall,
                "evidence_support_rate": support_rate,
                "supported_sentences": supported,
                "total_sentences": sentence_count,
            }

            print(f"Response: {response}")
            print(
                f"Answer-Question Relevance: {answer_relevance:.4f}"
                if answer_relevance is not None
                else "Answer-Question Relevance: N/A"
            )
            print(f"Token recall: {lexical_matched}/{lexical_total} ({lexical_recall:.4f})")
            print(f"Semantic answer points: {semantic_matched}/{semantic_total} ({semantic_recall:.4f})")
            print(f"NLI context coverage: {context_matched}/{context_total} ({context_recall:.4f})")
            print(f"Coverage gap: {context_recall - semantic_recall:.4f}")
            print(f"Retrieved evidence recall: {retrieved_matched}/{evidence_total} ({retrieved_recall:.4f})")
            print(f"Context evidence recall: {context_evidence_matched_count}/{context_evidence_total} ({context_evidence_recall:.4f})")
            print(f"Evidence support: {supported}/{sentence_count} ({support_rate:.4f})")
        else:
            if rejected:
                rejected_count += 1

            result = {
                "id": question_id,
                "answerable": False,
                "rejected": rejected,
            }

            print(f"Response: {response}")
            print(f"Correctly rejected: {rejected}")

        question_results.append(result)

    non_rejection_rate = answered_count / len(answerable_questions) if answerable_questions else 0.0
    token_recall = lexical_points_matched / total_expected_tokens if total_expected_tokens else 0.0
    semantic_point_recall = semantic_points_matched / total_expected_points if total_expected_points else 0.0
    context_recall = context_points_matched / total_context_points if total_context_points else 0.0
    retrieved_evidence_recall = (
        retrieved_evidence_matched / total_expected_evidence
        if total_expected_evidence else 0.0
    )
    context_evidence_recall = (
        context_evidence_matched / total_expected_evidence
        if total_expected_evidence else 0.0
    )
    evidence_support_rate = total_supported_sentences / total_answer_sentences if total_answer_sentences else 0.0
    answer_question_relevance = (
        total_answer_relevance_score / answer_relevance_count
        if answer_relevance_count
        else 0.0
    )
    rejection_rate = rejected_count / len(unanswerable_questions) if unanswerable_questions else 0.0

    print("\n" + "=" * 80)
    print("FINAL EVALUATION REPORT")
    print("=" * 80)
    print(f"Total questions: {len(questions)}")
    print(f"Answerable: {len(answerable_questions)}")
    print(f"Unanswerable: {len(unanswerable_questions)}")
    print(f"Non-rejection rate: {non_rejection_rate:.4f} ({answered_count}/{len(answerable_questions)})")
    print(f"Token recall: {token_recall:.4f} ({lexical_points_matched}/{total_expected_tokens} tokens)")
    print(
        f"Answer-Question Relevance: {answer_question_relevance:.4f} "
        f"({answer_relevance_count} answers)"
    )
    print(f"Semantic answer point recall: {semantic_point_recall:.4f} ({semantic_points_matched}/{total_expected_points} points)")
    print(
        f"NLI context coverage: {context_recall:.4f} "
        f"({context_points_matched}/{total_context_points} points)"
    )
    print(f"Coverage gap: {context_recall - semantic_point_recall:.4f}")
    print(
        f"Retrieved evidence recall: {retrieved_evidence_recall:.4f} "
        f"({retrieved_evidence_matched}/{total_expected_evidence} evidence items)"
    )
    print(
        f"Context evidence recall: {context_evidence_recall:.4f} "
        f"({context_evidence_matched}/{total_expected_evidence} evidence items)"
    )
    print(f"Evidence support rate: {evidence_support_rate:.4f} ({total_supported_sentences}/{total_answer_sentences})")
    print(f"Unanswerable rejection rate: {rejection_rate:.4f} ({rejected_count}/{len(unanswerable_questions)})")
    print("\n[QUESTION-LEVEL RESULTS]")
    for result in question_results:
        print(result)

    assert len(question_results) == len(questions)
    assert all(isinstance(result, dict) for result in question_results)
    assert semantic_point_recall >= 0.70
    assert context_recall >= 0.85
    assert context_recall - semantic_point_recall <= 0.15
    assert rejection_rate == 1.0