from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class SentenceStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"


class VerificationMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    NLI = "nli"
    NONE = "none"


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float

    def __post_init__(self):
        if any(
            not 0.0 <= score <= 1.0
            for score in (
                self.entailment,
                self.contradiction,
                self.neutral,
            )
        ):
            raise ValueError("NLI scores must be between 0 and 1.")


@dataclass
class SentenceVerification:
    sentence: str
    status: SentenceStatus
    method: VerificationMethod = VerificationMethod.NONE
    entailment_score: float | None = None
    contradiction_score: float | None = None
    neutral_score: float | None = None
    evidence: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class VerificationResult:
    status: VerificationStatus
    sentences: list[SentenceVerification] = field(default_factory=list)
    supported_count: int = 0
    unsupported_count: int = 0
    uncertain_count: int = 0
    contradicted_count: int = 0

    @property
    def is_supported(self) -> bool:
        return (
            bool(self.sentences)
            and self.status == VerificationStatus.SUPPORTED
            and all(
                item.status == SentenceStatus.SUPPORTED
                for item in self.sentences
            )
        )
