from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnswerabilityResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return self.status in {"SUFFICIENT", "PARTIAL"}


class AnswerabilityChecker:
    SYSTEM_PROMPT = (
        "Your task is to determine if the provided context contains ANY relevant "
        "information to address the user's query, even partially. Return 'True' if a "
        "partial or full answer can be formulated. Return 'False' ONLY IF the context "
        "is entirely irrelevant or contains absolutely no information related to the query."
    )

    def __init__(
        self,
        min_evidence_sentences: int = 1,
        allow_partial: bool = True,
        min_top_relevance: float | None = 0.0,
        min_aspect_coverage: float = 0.0,
        system_prompt: str | None = None,
    ):
        if min_evidence_sentences <= 0:
            raise ValueError("min_evidence_sentences must be greater than zero.")

        if not 0.0 <= min_aspect_coverage <= 1.0:
            raise ValueError("min_aspect_coverage must be between 0 and 1.")

        self.min_evidence_sentences = min_evidence_sentences
        self.allow_partial = allow_partial
        self.min_top_relevance = min_top_relevance
        self.min_aspect_coverage = min_aspect_coverage
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def assess(self, question: str, selection) -> AnswerabilityResult:
        if hasattr(selection, "sentences"):
            sentences = selection.sentences
            uncovered = list(
                getattr(selection, "uncovered_aspects", []) or []
            )
            aspects = list(getattr(selection, "aspects", []) or [])
            rejected = list(getattr(selection, "rejected", []) or [])
            top_relevance = getattr(selection, "top_relevance", None)
        else:
            sentences = selection if isinstance(selection, list) else []
            uncovered = []
            aspects = []
            rejected = []
            top_relevance = None

        texts = []

        for item in sentences:
            if isinstance(item, dict):
                sentence = item.get("sentence", "")
            else:
                sentence = item

            if isinstance(sentence, str) and sentence.strip():
                texts.append(sentence.strip())

        metrics = {
            "evidence_count": len(texts),
            "rejected_count": len(rejected),
            "top_relevance": top_relevance,
            "min_top_relevance": self.min_top_relevance,
        }

        if len(texts) < self.min_evidence_sentences:
            return AnswerabilityResult(
                "INSUFFICIENT",
                ["No verified evidence sentences were selected."],
                metrics,
            )

        # Relevance gate:
        # Evidence must be both verified and sufficiently relevant.
        if self.min_top_relevance is not None:
            if not isinstance(top_relevance, (int, float)):
                return AnswerabilityResult(
                    "INSUFFICIENT",
                    ["Relevance score is unavailable."],
                    metrics,
                )

            if top_relevance < self.min_top_relevance:
                return AnswerabilityResult(
                    "INSUFFICIENT",
                    ["Verified evidence is not relevant enough to the question."],
                    metrics,
                )

        metrics["aspect_count"] = len(aspects)

        if aspects:
            coverage = (
                len(aspects) - len(set(uncovered) & set(aspects))
            ) / len(aspects)

            metrics["aspect_coverage"] = coverage
            metrics["uncovered_aspects"] = uncovered

            if coverage <= 0.0 or coverage < self.min_aspect_coverage:
                return AnswerabilityResult(
                    "INSUFFICIENT",
                    ["No question aspect is covered by verified evidence."],
                    metrics,
                )

        if uncovered:
            status = (
                "PARTIAL"
                if self.allow_partial
                else "INSUFFICIENT"
            )

            metrics["uncovered_aspects"] = uncovered

            return AnswerabilityResult(
                status,
                ["Some question aspects are not covered."],
                metrics,
            )

        return AnswerabilityResult(
            "SUFFICIENT",
            ["Selected evidence passed relevance and verification gates."],
            metrics,
        )

    @staticmethod
    def is_answerable(selected_sentences: list[str]) -> bool:
        return any(
            isinstance(sentence, str) and sentence.strip()
            for sentence in selected_sentences
        )
