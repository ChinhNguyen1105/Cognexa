import re


class QueryAnalyzer:
    """
    Classify questions and determine whether they require decomposition.
    """

    QUESTION_TYPES = {
        "COMPARISON": "comparison",
        "CAUSE_EFFECT": "cause_effect",
        "MULTI_ASPECT": "multi_aspect",
        "RELATIONSHIP": "relationship",
        "SIMPLE": "simple",
    }

    CLASSIFICATION_PATTERNS = {
        "comparison": [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifferences?\s+between\b",
            r"\badvantages?\s+and\s+disadvantages?\b",
            r"\bpros\s+and\s+cons\b",
            r"\bstrengths?\s+and\s+weaknesses\b",
            r"\btrade[- ]offs?\b",
            r"\bso\s+sánh\b",
            r"\bkhác\s+nhau\b",
            r"\bưu\s+điểm\s+và\s+nhược\s+điểm\b",
        ],
        "cause_effect": [
            r"\bcauses?\s+and\s+consequences?\b",
            r"\bcauses?\s+and\s+effects?\b",
            r"\breasons?\s+and\s+consequences?\b",
            r"\bnguyên\s+nhân\s+và\s+hậu\s+quả\b",
        ],
        "multi_aspect": [
            r"\bwhat\s+.*\s+and\s+how\b",
            r"\bhow\s+.*\s+and\s+why\b",
            r"\bwhat\s+.*\s+and\s+why\b",
            r"\bexplain\s+.*\s+as\s+well\s+as\b",
        ],
        "relationship": [
            r"\brelationship\s+between\b",
            r"\bconnection\s+between\b",
            r"\bhow\s+.*\s+relate\s+to\b",
            r"\brelationship\s+of\b",
            r"\bmối\s+quan\s+hệ\b",
            r"\bliên\s+hệ\s+giữa\b",
        ],
    }

    def classify(self, query: str) -> str:
        """
        Classify a question using lightweight pattern matching.

        Returns:
            comparison, cause_effect, multi_aspect,
            relationship, or simple.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        normalized_query = query.lower().strip()

        # More specific question types must be checked first.
        for question_type in (
            "comparison",
            "cause_effect",
            "relationship",
            "multi_aspect",
        ):
            patterns = self.CLASSIFICATION_PATTERNS[question_type]

            if any(
                re.search(pattern, normalized_query)
                for pattern in patterns
            ):
                return question_type

        # Multiple explicit questions may require decomposition.
        if normalized_query.count("?") > 1:
            return "multi_aspect"

        return "simple"

    def is_complex(self, query: str) -> bool:
        """
        Determine whether a question may contain multiple information needs.

        This is a heuristic, not a semantic classifier.
        """
        return self.classify(query) != "simple"
