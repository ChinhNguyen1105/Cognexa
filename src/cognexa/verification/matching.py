from cognexa.verification.preprocessing import TextPreprocessor


class EvidenceMatcher:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def exact_match(
        self,
        sentence: str,
        evidence: list[str],
    ) -> bool:
        normalized_sentence = self.preprocessor.normalize(
            sentence
        )

        if not normalized_sentence:
            return False

        return any(
            normalized_sentence == self.preprocessor.normalize(item)
            for item in evidence
            if isinstance(item, str)
        )

    def find_exact_evidence(
        self,
        sentence: str,
        evidence: list[str],
    ) -> list[str]:
        normalized_sentence = self.preprocessor.normalize(
            sentence
        )

        if not normalized_sentence:
            return []

        return [
            item
            for item in evidence
            if isinstance(item, str)
            and self.preprocessor.normalize(item) == normalized_sentence
        ]

    def find_candidate_evidence(
        self,
        sentence: str,
        evidence: list[str],
        min_overlap: float = 0.3,
    ) -> list[str]:
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError(
                "min_overlap must be between 0 and 1."
            )

        sentence_tokens = set(
            self.preprocessor.normalize(sentence).split()
        )

        if not sentence_tokens:
            return []

        matches = []

        for position, item in enumerate(evidence):
            if not isinstance(item, str):
                continue

            evidence_tokens = set(
                self.preprocessor.normalize(item).split()
            )

            if not evidence_tokens:
                continue

            overlap = (
                len(sentence_tokens & evidence_tokens)
                / len(sentence_tokens)
            )

            if overlap >= min_overlap:
                matches.append((overlap, position, item))

        # Best overlap first (ties keep document order), so that a caller
        # truncating to candidate_limit keeps the most relevant evidence
        # instead of simply the earliest sentences.
        matches.sort(key=lambda match: (-match[0], match[1]))

        return [item for _, _, item in matches]
