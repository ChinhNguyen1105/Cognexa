
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
        "PREVENTION": "prevention",
        "LIST": "list",
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
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bso\s+sánh\b",
            r"\bkhác\s+nhau\b",
            r"\bưu\s+điểm\s+và\s+nhược\s+điểm\b",
        ],
        "cause_effect": [
            r"\bcauses?\s+and\s+consequences?\b",
            r"\bcauses?\s+and\s+effects?\b",
            r"\breasons?\s+and\s+consequences?\b",
            r"\bwhat\s+happens\s+when\b",
            r"\bwhat\s+are\s+the\s+effects?\b",
            r"\bwhy\s+does\b",
            r"\bwhat\s+causes?\b",
            r"\bnguyên\s+nhân\s+và\s+hậu\s+quả\b",
        ],
        "multi_aspect": [
            r"\bwhat\s+.*\s+and\s+how\b",
            r"\bhow\s+.*\s+and\s+why\b",
            r"\bwhat\s+.*\s+and\s+why\b",
            r"\bexplain\s+.*\s+as\s+well\s+as\b",
            r"\bwhat\s+is\s+.*\s+and\s+how\b",
            r"\bhow\s+does\s+.*\s+and\s+when\b",
        ],
        "relationship": [
            r"\brelationship\s+between\b",
            r"\bconnection\s+between\b",
            r"\bhow\s+.*\s+relate\s+to\b",
            r"\brelationship\s+of\b",
            r"\bhow\s+are\s+.*\s+and\s+.*\s+related\b",
            r"\bwhat\s+is\s+the\s+link\s+between\b",
            r"\bmối\s+quan\s+hệ\b",
            r"\bliên\s+hệ\s+giữa\b",
            r"\bgiữa\s+.*\s+và\s+.*\b",
        ],
        "prevention": [
            r"\bhow\s+can\s+.*\s+(?:be\s+)?(?:reduced|minimized|mitigated|prevented|avoided)\b",
            r"\bhow\s+to\s+(?:reduce|minimize|mitigate|prevent|avoid)\b",
            r"\bhow\s+can\s+(?:we\s+)?(?:reduce|minimize|mitigate|prevent|avoid)\b",
            r"\bwhat\s+(?:methods?|ways?|techniques?|strategies?|approaches?)\s+can\s+(?:we\s+)?(?:reduce|minimize|mitigate|prevent|avoid)\b",
            r"\bwhat\s+(?:methods?|ways?|techniques?|strategies?|approaches?)\s+can\s+.*\s+(?:reduce|minimize|mitigate|prevent|avoid)\b",
            r"\b(?:methods?|ways?|techniques?|strategies?|approaches?)\s+to\s+(?:reduce|minimize|mitigate|prevent|avoid)\b",
        ],
        "list": [
            r"\bwhat\s+are\s+(?:the\s+)?(?:other\s+)?(?:main\s+|common\s+|key\s+)?(?:factors?|causes?|consequences?|effects?|methods?|ways?|techniques?|strategies?|approaches?|types?|examples?)\b",
            r"\blist\s+(?:the\s+)?(?:other\s+)?(?:main\s+|common\s+|key\s+)?(?:factors?|causes?|consequences?|effects?|methods?|ways?|techniques?|strategies?|approaches?|types?|examples?)\b",
            r"\bwhat\s+other\s+(?:factors?|causes?|methods?|ways?|techniques?|strategies?|approaches?)\b",
        ],
    }

    def classify(self, query: str) -> str:
        """
        Classify a question using lightweight pattern matching.

        Returns:
            comparison, cause_effect, relationship, multi_aspect,
            prevention, list, or simple.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        normalized_query = query.lower().strip()

        for question_type in (
            "comparison",
            "cause_effect",
            "relationship",
            "multi_aspect",
            "prevention",
            "list",
        ):
            patterns = self.CLASSIFICATION_PATTERNS[question_type]

            if any(
                re.search(pattern, normalized_query)
                for pattern in patterns
            ):
                return question_type

        if normalized_query.count("?") > 1:
            return "multi_aspect"

        return "simple"

    def is_complex(self, query: str) -> bool:
        """
        Determine whether a question may contain multiple information needs.

        This is a heuristic, not a semantic classifier.
        """
        return self.classify(query) != "simple"
