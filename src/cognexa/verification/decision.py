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
        entailment_threshold: float = 0.75,
        contradiction_threshold: float = 0.75,
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

        best_entailment = max(
            candidates,
            key=lambda item: item[1].entailment,
        )
        best_contradiction = max(
            candidates,
            key=lambda item: item[1].contradiction,
        )

        entailment = best_entailment[1].entailment
        contradiction = best_contradiction[1].contradiction

        support_found = entailment >= self.entailment_threshold
        contradiction_found = (
            contradiction >= self.contradiction_threshold
        )

        if support_found and contradiction_found:
            status = SentenceStatus.UNCERTAIN
            selected_evidence = [
                best_entailment[0],
                best_contradiction[0],
            ]
            reason = "Conflicting evidence found."

        elif contradiction_found:
            status = SentenceStatus.CONTRADICTED
            selected_evidence = [best_contradiction[0]]
            reason = "Evidence contradicts the answer."

        elif support_found:
            status = SentenceStatus.SUPPORTED
            selected_evidence = [best_entailment[0]]
            reason = "Evidence entails the answer."

        elif max(
            entailment,
            contradiction,
        ) >= self.uncertainty_threshold:
            status = SentenceStatus.UNCERTAIN
            selected_evidence = [
                max(
                    candidates,
                    key=lambda item: max(
                        item[1].entailment,
                        item[1].contradiction,
                    ),
                )[0]
            ]
            reason = "NLI evidence is inconclusive."

        else:
            status = SentenceStatus.UNSUPPORTED
            selected_evidence = []
            reason = "No sufficient entailment found."

        return SentenceVerification(
            sentence=sentence,
            status=status,
            method=VerificationMethod.NLI,
            entailment_score=entailment,
            contradiction_score=contradiction,
            neutral_score=max(
                item[1].neutral for item in candidates
            ),
            evidence=list(dict.fromkeys(selected_evidence)),
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
