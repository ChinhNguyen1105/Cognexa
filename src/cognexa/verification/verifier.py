from cognexa.verification.decision import VerificationDecision
from cognexa.verification.matching import EvidenceMatcher
from cognexa.verification.nli_model import NLIModel
from cognexa.verification.preprocessing import TextPreprocessor
from cognexa.verification.schemas import (
    SentenceStatus,
    SentenceVerification,
    VerificationResult,
)


class AnswerVerifier:
    def __init__(
        self,
        nli_model: NLIModel | None = None,
        matcher: EvidenceMatcher | None = None,
        decision: VerificationDecision | None = None,
        candidate_limit: int = 5,
        min_overlap: float = 0.3,
    ):
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than 0.")

        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_overlap must be between 0 and 1.")

        # The NLI model is loaded lazily so constructing an AnswerVerifier
        # (for example as the Assistant default) does not load a large model.
        self._nli_model = nli_model
        self.matcher = matcher or EvidenceMatcher()
        self.decision = decision or VerificationDecision()
        self.preprocessor = TextPreprocessor()
        self.candidate_limit = candidate_limit
        self.min_overlap = min_overlap

    @property
    def nli_model(self) -> NLIModel:
        if self._nli_model is None:
            self._nli_model = NLIModel()

        return self._nli_model

    @nli_model.setter
    def nli_model(self, value: NLIModel) -> None:
        self._nli_model = value

    def _prepare_evidence(self, evidence: list[str]) -> list[str]:
        sentences = []

        for item in evidence:
            # split_evidence also splits on line breaks and drops
            # "[Document N | ...]" headers, matching AnswerExtractor units.
            sentences.extend(
                self.preprocessor.split_evidence(item)
            )

        unique_sentences = []
        seen = set()

        for sentence in sentences:
            normalized = self.preprocessor.normalize(sentence)

            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)

        return unique_sentences

    def verify(
        self,
        answer: str,
        evidence: list[str],
    ) -> VerificationResult:

        if not isinstance(answer, str):
            raise TypeError("answer must be a string.")

        if not isinstance(evidence, list):
            raise TypeError("evidence must be a list.")

        if any(not isinstance(item, str) for item in evidence):
            raise TypeError("Every evidence item must be a string.")

        sentences = self.preprocessor.split_sentences(answer)

        if not sentences:
            return self._build_result([])

        if not evidence:
            return self._build_result([
                SentenceVerification(
                    sentence=sentence,
                    status=SentenceStatus.UNSUPPORTED,
                    reason="No evidence was provided.",
                )
                for sentence in sentences
            ])

        evidence_sentences = self._prepare_evidence(evidence)

        sentence_candidates = []
        premises = []
        hypotheses = []

        for sentence in sentences:
            exact_matches = self.matcher.find_exact_evidence(
                sentence,
                evidence_sentences,
            )

            if exact_matches:
                candidates = exact_matches
            else:
                candidates = self.matcher.find_candidate_evidence(
                    sentence,
                    evidence_sentences,
                    min_overlap=self.min_overlap,
                )[:self.candidate_limit]

            sentence_candidates.append(candidates)

            for candidate in candidates:
                premises.append(candidate)
                hypotheses.append(sentence)

        scores = (
            self.nli_model.predict_batch(premises, hypotheses)
            if premises
            else []
        )

        results = []
        score_index = 0

        for sentence, candidates in zip(
            sentences,
            sentence_candidates,
        ):
            if not candidates:
                results.append(
                    SentenceVerification(
                        sentence=sentence,
                        status=SentenceStatus.UNSUPPORTED,
                        reason="No relevant evidence candidates found.",
                    )
                )
                continue

            candidate_scores = scores[
                score_index:score_index + len(candidates)
            ]
            score_index += len(candidates)

            results.append(
                self.decision.evaluate_candidates(
                    sentence,
                    list(zip(candidates, candidate_scores)),
                )
            )

        return self._build_result(results)

    def _build_result(
        self,
        sentences: list[SentenceVerification],
    ) -> VerificationResult:

        supported = sum(
            item.status == SentenceStatus.SUPPORTED
            for item in sentences
        )
        unsupported = sum(
            item.status == SentenceStatus.UNSUPPORTED
            for item in sentences
        )
        uncertain = sum(
            item.status == SentenceStatus.UNCERTAIN
            for item in sentences
        )
        contradicted = sum(
            item.status == SentenceStatus.CONTRADICTED
            for item in sentences
        )

        return VerificationResult(
            status=self.decision.aggregate(sentences),
            sentences=sentences,
            supported_count=supported,
            unsupported_count=unsupported,
            uncertain_count=uncertain,
            contradicted_count=contradicted,
        )
