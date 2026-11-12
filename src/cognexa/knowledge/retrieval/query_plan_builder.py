from __future__ import annotations

from typing import Any, Sequence

from cognexa.knowledge.query.query_analyzer import QueryAnalyzer

from .contracts.query import QueryPlan, QueryType, RetrievalQuery


class QueryPlanBuilder:
    """
    Convert QuestionDecomposer output into a structured QueryPlan.
    """

    REQUIRED_ASPECTS = {
        "comparison",
        "causes",
        "consequences",
        "effects",
        "reasons",
        "definition",
        "process",
        "benefits",
        "limitations",
        "similarities",
        "differences",
        "relationship",
        "interaction",
        "dependency",
    }

    def __init__(
        self,
        analyzer: QueryAnalyzer | None = None,
    ):
        self.analyzer = analyzer or QueryAnalyzer()

    def build(
        self,
        original_question: str,
        decomposed_questions: Sequence[dict[str, Any]],
    ) -> QueryPlan:
        if not isinstance(original_question, str):
            raise TypeError("original_question must be a string.")

        original_question = original_question.strip()

        if not original_question:
            raise ValueError("original_question cannot be empty.")

        if not isinstance(decomposed_questions, (list, tuple)):
            raise TypeError("decomposed_questions must be a list or tuple.")

        intent = self.analyzer.classify(original_question)

        retrieval_queries = [
            RetrievalQuery(
                query_id="original",
                text=original_question,
                query_type=QueryType.ORIGINAL,
                aspect="original",
                required=True,
            )
        ]

        sub_questions = []
        seen = {original_question.casefold()}

        for item in decomposed_questions:
            if not isinstance(item, dict):
                raise TypeError(
                    "Every decomposed question must be a dictionary."
                )

            aspect = item.get("aspect")
            question = item.get("question")

            if not isinstance(aspect, str):
                raise TypeError("Every aspect must be a string.")

            if not isinstance(question, str):
                raise TypeError("Every question must be a string.")

            aspect = aspect.strip().casefold()
            question = question.strip()

            if not aspect or not question:
                continue

            normalized = question.casefold()

            if normalized in seen:
                continue

            seen.add(normalized)

            query_id = f"sub_question_{len(sub_questions) + 1}"
            required = (
                intent != "simple"
                or aspect in self.REQUIRED_ASPECTS
            )

            retrieval_queries.append(
                RetrievalQuery(
                    query_id=query_id,
                    text=question,
                    query_type=QueryType.SUB_QUESTION,
                    aspect=aspect,
                    required=required,
                )
            )

            sub_questions.append(question)

        normalized_query = " ".join(
            original_question.casefold().split()
        )

        return QueryPlan(
            original_query=original_question,
            normalized_query=normalized_query,
            intent=intent,
            retrieval_queries=retrieval_queries,
            sub_questions=sub_questions,
        )
