import re
from dataclasses import dataclass, field

from cognexa.knowledge.validation.answer_verifier import AnswerVerifier


@dataclass
class SelectionResult:
    sentences: list[dict] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)
    uncovered_aspects: list[str] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [
            item["sentence"]
            for item in self.sentences
        ]


class AnswerSelector:
    """
    Select supported, relevant and non-redundant evidence.

    Evidence support and aspect coverage are treated as
    separate dimensions.
    """

    def __init__(
        self,
        answer_verifier: AnswerVerifier,
        min_relevance_score: float | None = None,
        max_sentences: int = 5,
        redundancy_threshold: float = 0.85,
    ):
        if max_sentences <= 0:
            raise ValueError(
                "max_sentences must be greater than zero."
            )

        if not 0 <= redundancy_threshold <= 1:
            raise ValueError(
                "redundancy_threshold must be between 0 and 1."
            )

        self.answer_verifier = answer_verifier
        self.min_relevance_score = min_relevance_score
        self.max_sentences = max_sentences
        self.redundancy_threshold = redundancy_threshold

    def select(
        self,
        candidates: list[dict],
        context: str,
    ) -> list[str]:
        return self.select_detailed(
            candidates=candidates,
            context=context,
        ).texts

    def select_detailed(
        self,
        candidates: list[dict],
        context: str,
        question: str | None = None,
    ) -> SelectionResult:

        if (
            not candidates
            or not isinstance(context, str)
            or not context.strip()
        ):
            return SelectionResult()

        aspects = []

        for candidate in candidates:
            for aspect in candidate.get("aspect_scores", {}):
                if aspect not in aspects:
                    aspects.append(aspect)

        ranked_candidates = sorted(
            candidates,
            key=lambda item: item.get(
                "question_score",
                item.get("score", float("-inf")),
            ),
            reverse=True,
        )

        selected = []
        covered_aspects = set()

        for candidate in ranked_candidates:

            if len(selected) >= self.max_sentences:
                break

            sentence = candidate.get("sentence")

            if (
                not isinstance(sentence, str)
                or not sentence.strip()
            ):
                continue

            score = candidate.get("question_score")

            if score is None:
                score = candidate.get(
                    "score",
                    float("-inf"),
                )

            if (
                self.min_relevance_score is not None
                and score < self.min_relevance_score
            ):
                continue

            verification = self.answer_verifier.verify_sentence(
                context=context,
                hypothesis=sentence,
            )

            if (
                verification.get("status")
                != AnswerVerifier.SUPPORTED
            ):
                continue

            if self._is_redundant(
                sentence,
                [
                    item["sentence"]
                    for item in selected
                ],
            ):
                continue

            # IMPORTANT:
            # aspect_scores only represent ranking signals.
            # They do not prove that the sentence answers an aspect.
            #
            # Actual coverage must be explicitly supplied by
            # the extraction/relevance analysis stage.
            explicit_coverage = candidate.get(
                "covered_aspects",
                [],
            )

            if not isinstance(explicit_coverage, list):
                explicit_coverage = []

            selected.append({
                **candidate,
                "sentence": sentence,
                "verification": verification,
            })

            covered_aspects.update(
                aspect
                for aspect in explicit_coverage
                if aspect in aspects
            )

        uncovered_aspects = [
            aspect
            for aspect in aspects
            if aspect not in covered_aspects
        ]

        return SelectionResult(
            sentences=selected,
            aspects=aspects,
            uncovered_aspects=uncovered_aspects,
        )

    def _is_redundant(
        self,
        sentence: str,
        selected: list[str],
    ) -> bool:
        return any(
            self._jaccard_similarity(
                sentence,
                existing,
            ) >= self.redundancy_threshold
            for existing in selected
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(
            re.findall(
                r"\b\w+\b",
                text.casefold(),
            )
        )

    @classmethod
    def _jaccard_similarity(
        cls,
        first: str,
        second: str,
    ) -> float:
        first_tokens = cls._tokenize(first)
        second_tokens = cls._tokenize(second)

        if not first_tokens or not second_tokens:
            return 0.0

        return (
            len(first_tokens & second_tokens)
            / len(first_tokens | second_tokens)
        )
