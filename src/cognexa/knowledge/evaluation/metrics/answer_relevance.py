
import math


def calculate_answer_question_relevance(
    question: str,
    answer: str,
    reranker,
) -> float | None:
    """
    Calculate the relevance score between a question and a final answer.

    Uses the existing CrossEncoderReranker.

    Returns:
        Raw Cross-Encoder relevance score, or None if the answer is empty.

    Note:
        The score is not a probability and is not normalized to [0, 1].
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    if not answer.strip():
        return None

    if not hasattr(reranker, "rerank"):
        raise TypeError("reranker must provide a rerank method.")

    results = reranker.rerank(
        query=question,
        candidates=[answer],
    )

    if not results:
        raise RuntimeError(
            "Reranker returned no relevance score for a non-empty answer."
        )

    score = results[0].get("score")

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("Reranker returned an invalid relevance score.")

    score = float(score)

    if not math.isfinite(score):
        raise RuntimeError("Reranker returned a non-finite relevance score.")

    return score
