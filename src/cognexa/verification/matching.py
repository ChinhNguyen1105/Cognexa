from cognexa.verification.preprocessing import TextPreprocessor


class EvidenceMatcher:
    STOPWORDS = {
        "a", "an", "the",
        "is", "am", "are", "be", "been", "being",
        "do", "does", "did",
        "have", "has", "had",
        "can", "could", "may", "might", "must",
        "shall", "should", "will", "would",
        "of", "at", "by", "for", "from",
        "in", "into", "on", "onto", "to", "with",
        "about", "after", "before", "between",
        "and", "or", "but", "if", "because",
        "as", "than", "that", "this", "these", "those",
        "it", "its", "they", "them", "their",
        "he", "she", "we", "you", "i",
        "not", "no",
    }

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def _content_tokens(self, text: str) -> set[str]:
        normalized = self.preprocessor.normalize(text)

        return {
            token
            for token in normalized.split()
            if token not in self.STOPWORDS
        }

    def exact_match(self, sentence: str, evidence: list[str]) -> bool:
        normalized_sentence = self.preprocessor.normalize(sentence)

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
        normalized_sentence = self.preprocessor.normalize(sentence)

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
        min_overlap: float = 0.15,
    ) -> list[str]:
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_overlap must be between 0 and 1.")

        sentence_tokens = self._content_tokens(sentence)

        if not sentence_tokens:
            return []

        matches = []

        for position, item in enumerate(evidence):
            if not isinstance(item, str):
                continue

            evidence_tokens = self._content_tokens(item)

            if not evidence_tokens:
                continue

            overlap = (
                len(sentence_tokens & evidence_tokens)
                / len(sentence_tokens)
            )

            if overlap >= min_overlap:
                matches.append((overlap, position, item))

        matches.sort(key=lambda match: (-match[0], match[1]))

        return [
            item
            for _, _, item in matches
        ]
