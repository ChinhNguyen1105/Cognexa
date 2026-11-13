from cognexa.verification.schemas import (
    NLIScores,
    SentenceStatus,
    SentenceVerification,
    VerificationMethod,
    VerificationStatus,
)


class VerificationDecision:
    def __init__(
        self,
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.70,
        uncertainty_threshold: float = 0.50,
    ):
        thresholds = (
            entailment_threshold,
            contradiction_threshold,
            uncertainty_threshold,
        )

        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("Thresholds must be between 0 and 1.")

        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def evaluate(
        self,
        sentence: str,
        scores: NLIScores,
        evidence: list[str],
    ) -> SentenceVerification:
        return self.evaluate_candidates(
            sentence,
            [(item, scores) for item in evidence],
        )

    def evaluate_candidates(
        self,
        sentence: str,
        candidates: list[tuple[str, NLIScores]],
    ) -> SentenceVerification:

        if not candidates:
            return SentenceVerification(
                sentence=sentence,
                status=SentenceStatus.UNSUPPORTED,
                reason="No evidence candidates provided.",
            )

        # Each candidate must be evaluated independently.
        conflicting = [
            (evidence, scores)
            for evidence, scores in candidates
            if (
                scores.entailment >= self.entailment_threshold
                and scores.contradiction >= self.contradiction_threshold
            )
        ]

        if conflicting:
            selected_evidence, selected_scores = max(
                conflicting,
                key=lambda item: min(
                    item[1].entailment,
                    item[1].contradiction,
                ),
            )

            status = SentenceStatus.UNCERTAIN
            evidence = [selected_evidence]
            reason = "Conflicting signals found in the same evidence candidate."

        else:
            supporting = [
                (evidence, scores)
                for evidence, scores in candidates
                if scores.entailment >= self.entailment_threshold
            ]

            contradicting = [
                (evidence, scores)
                for evidence, scores in candidates
                if scores.contradiction >= self.contradiction_threshold
            ]

            inconclusive = [
                (evidence, scores)
                for evidence, scores in candidates
                if max(
                    scores.entailment,
                    scores.contradiction,
                ) >= self.uncertainty_threshold
            ]

            if supporting:
                selected_evidence, selected_scores = max(
                    supporting,
                    key=lambda item: item[1].entailment,
                )

                status = SentenceStatus.SUPPORTED
                evidence = [selected_evidence]
                reason = "Evidence entails the answer."

            elif contradicting:
                selected_evidence, selected_scores = max(
                    contradicting,
                    key=lambda item: item[1].contradiction,
                )

                status = SentenceStatus.CONTRADICTED
                evidence = [selected_evidence]
                reason = "Evidence contradicts the answer."

            elif inconclusive:
                selected_evidence, selected_scores = max(
                    inconclusive,
                    key=lambda item: max(
                        item[1].entailment,
                        item[1].contradiction,
                    ),
                )

                status = SentenceStatus.UNCERTAIN
                evidence = [selected_evidence]
                reason = "NLI evidence is inconclusive."

            else:
                selected_scores = None
                status = SentenceStatus.UNSUPPORTED
                evidence = []
                reason = "No sufficient entailment found."

        return SentenceVerification(
            sentence=sentence,
            status=status,
            method=VerificationMethod.NLI,
            entailment_score=(
                selected_scores.entailment
                if selected_scores is not None
                else max(item[1].entailment for item in candidates)
            ),
            contradiction_score=(
                selected_scores.contradiction
                if selected_scores is not None
                else max(item[1].contradiction for item in candidates)
            ),
            neutral_score=max(
                item[1].neutral for item in candidates
            ),
            evidence=list(dict.fromkeys(evidence)),
            reason=reason,
        )

    @staticmethod
    def aggregate(
        sentences: list[SentenceVerification],
    ) -> VerificationStatus:

        if not sentences:
            return VerificationStatus.UNSUPPORTED

        statuses = [item.status for item in sentences]

        if all(
            status == SentenceStatus.SUPPORTED
            for status in statuses
        ):
            return VerificationStatus.SUPPORTED

        if all(
            status == SentenceStatus.UNSUPPORTED
            for status in statuses
        ):
            return VerificationStatus.UNSUPPORTED

        if all(
            status == SentenceStatus.CONTRADICTED
            for status in statuses
        ):
            return VerificationStatus.CONTRADICTED

        return VerificationStatus.PARTIALLY_SUPPORTED