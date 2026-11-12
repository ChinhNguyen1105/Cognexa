from __future__ import annotations

from collections.abc import Sequence

from cognexa.knowledge.retrieval.contracts import (
    CoverageReport,
    Evidence,
    QueryPlan,
    SubQuestionCoverage,
)


class CoverageAnalyzer:
    """Analyze how well retrieved evidence covers a query plan."""

    def evaluate(
        self,
        query_plan: QueryPlan,
        evidence: Sequence[Evidence],
    ) -> CoverageReport:
        if not isinstance(query_plan, QueryPlan):
            raise TypeError("query_plan must be a QueryPlan")

        if isinstance(evidence, (str, bytes)):
            raise TypeError("evidence must be a sequence of Evidence")

        evidence_list = list(evidence)

        if not all(isinstance(item, Evidence) for item in evidence_list):
            raise TypeError("All evidence items must be Evidence instances")

        results: list[SubQuestionCoverage] = []

        total_weight = 0.0
        covered_weight = 0.0

        for query in query_plan.retrieval_queries:
            related_evidence = [
                item
                for item in evidence_list
                if query.query_id in item.related_query_ids
            ]

            is_covered = bool(related_evidence)
            coverage_score = 1.0 if is_covered else 0.0

            total_weight += query.weight

            if is_covered:
                covered_weight += query.weight

            results.append(
                SubQuestionCoverage(
                    query_id=query.query_id,
                    question=query.text,
                    coverage_score=coverage_score,
                    covered=is_covered,
                    evidence_ids=list(
                        dict.fromkeys(
                            item.evidence_id
                            for item in related_evidence
                        )
                    ),
                    missing_information=(
                        None
                        if is_covered
                        else "No evidence retrieved for this query."
                    ),
                    required=query.required,
                )
            )

        overall_score = (
            covered_weight / total_weight
            if total_weight
            else 0.0
        )

        return CoverageReport(
            overall_score=overall_score,
            sub_question_results=results,
        )
