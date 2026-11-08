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
    """Assess whether selected evidence provides full, partial, or no answer."""

    def __init__(self, min_evidence_sentences: int = 1, allow_partial: bool = True):
        if min_evidence_sentences <= 0:
            raise ValueError("min_evidence_sentences must be greater than zero.")
        self.min_evidence_sentences = min_evidence_sentences
        self.allow_partial = allow_partial

    def assess(self, question: str, selection) -> AnswerabilityResult:
        if hasattr(selection, "sentences"):
            sentences = selection.sentences
            uncovered = list(getattr(selection, "uncovered_aspects", []))
            aspects = list(getattr(selection, "aspects", []))
        else:
            sentences = selection if isinstance(selection, list) else []
            uncovered, aspects = [], []
        texts = [item.get("sentence", "") if isinstance(item, dict) else item for item in sentences]
        valid = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        if len(valid) < self.min_evidence_sentences:
            return AnswerabilityResult("INSUFFICIENT", ["No verified evidence sentences were selected."], {"evidence_count": len(valid)})
        if uncovered:
            status = "PARTIAL" if self.allow_partial else "INSUFFICIENT"
            return AnswerabilityResult(status, ["Some question aspects are not covered."], {"evidence_count": len(valid), "aspect_count": len(aspects), "uncovered_aspects": uncovered})
        return AnswerabilityResult("SUFFICIENT", ["Selected evidence passed verification."], {"evidence_count": len(valid), "aspect_count": len(aspects)})

    @staticmethod
    def is_answerable(selected_sentences: list[str]) -> bool:
        return any(isinstance(sentence, str) and sentence.strip() for sentence in selected_sentences)
