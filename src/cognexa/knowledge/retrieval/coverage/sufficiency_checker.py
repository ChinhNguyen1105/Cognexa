from __future__ import annotations

import math

from cognexa.knowledge.retrieval.contracts import (
    CoverageReport,
    SufficiencyDecision,
)


class SufficiencyChecker:
    """Decide whether retrieved evidence is sufficient."""

    def __init__(self, min_coverage: float = 0.7) -> None:
        if isinstance(min_coverage, bool) or not isinstance(
            min_coverage, (int, float)
        ):
            raise TypeError("min_coverage must be a number")

        if not math.isfinite(min_coverage):
            raise ValueError("min_coverage must be finite")

        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0 and 1")

        self.min_coverage = float(min_coverage)

    def check(self, report: CoverageReport) -> SufficiencyDecision:
        if not isinstance(report, CoverageReport):
            raise TypeError("report must be a CoverageReport")

        missing_query_ids = tuple(
            result.query_id
            for result in report.sub_question_results
            if not result.covered
        )

        missing_required_ids = tuple(
            result.query_id
            for result in report.sub_question_results
            if result.required and not result.covered
        )

        coverage_met = report.overall_score >= self.min_coverage
        required_queries_met = not missing_required_ids
        sufficient = coverage_met and required_queries_met

        if sufficient:
            return SufficiencyDecision(
                sufficient=True,
                reason=(
                    f"Coverage score {report.overall_score:.2f} "
                    f"meets the minimum threshold "
                    f"{self.min_coverage:.2f}, and all required "
                    f"queries are covered."
                ),
            )

        reasons = []

        if not coverage_met:
            reasons.append(
                f"Coverage score {report.overall_score:.2f} "
                f"is below the minimum threshold "
                f"{self.min_coverage:.2f}"
            )

        if missing_required_ids:
            reasons.append(
                "Required queries are not covered: "
                f"{', '.join(missing_required_ids)}"
            )

        reason = "; ".join(reasons) + "."

        return SufficiencyDecision(
            sufficient=False,
            missing_query_ids=missing_query_ids,
            reason=reason,
        )
