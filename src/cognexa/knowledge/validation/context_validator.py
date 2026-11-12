import re


class ContextValidator:
    """
    Validate whether retrieved documents are semantically
    relevant to the user's question.
    """

    def __init__(self, max_distance: float = 0.55):
        if max_distance < 0:
            raise ValueError(
                "max_distance must be greater than or equal to 0."
            )

        self.max_distance = max_distance

    @staticmethod
    def _normalize_text(text: str) -> set[str]:
        if not isinstance(text, str):
            return set()
        return {
            token
            for token in re.findall(r"\b[\w-]+\b", text.casefold())
            if token and token not in {"the", "a", "an", "how", "what", "why", "when", "where", "which", "who", "is", "are", "do", "does", "did", "of", "to", "in", "on", "for", "and", "or", "with", "between", "by"}
        }

    @classmethod
    def _question_overlap(cls, question: str, document: dict) -> float:
        if not question or not isinstance(question, str):
            return 0.0

        content = (
            document.get("content")
            or document.get("text")
            or document.get("chunk", "")
            or ""
        )
        if not isinstance(content, str) or not content.strip():
            return 0.0

        question_tokens = cls._normalize_text(question)
        content_tokens = cls._normalize_text(content)
        if not question_tokens or not content_tokens:
            return 0.0

        overlap = len(question_tokens & content_tokens)
        if not overlap:
            return 0.0

        return overlap / max(len(question_tokens), 1)

    def is_relevant(
        self,
        documents: list[dict],
        question: str | None = None,
    ) -> bool:
        if not documents:
            print("[DEBUG VALIDATOR] No documents received.")
            return False

        print("\\n[DEBUG VALIDATOR]")
        print(f"Max distance: {self.max_distance}")

        any_relevant = False

        for index, document in enumerate(documents, start=1):
            distance = document.get("distance")
            overlap = (
                self._question_overlap(question, document)
                if question else 0.0
            )

            print(
                f"Document {index} | "
                f"Distance: {distance} | "
                f"Threshold: {self.max_distance} | "
                f"Overlap: {overlap:.2f}"
            )

            document_relevant = False

            if distance is not None:
                try:
                    document_relevant = (
                        float(distance) <= self.max_distance
                    )
                except (TypeError, ValueError):
                    document_relevant = False

            if not document_relevant and question is not None:
                threshold = 0.15 if distance is None else 0.20
                document_relevant = overlap >= threshold

            if document_relevant:
                any_relevant = True
                print(" -> Relevant")
            else:
                print(" -> Not relevant")

        if any_relevant:
            print("[DEBUG VALIDATOR] At least one relevant document found.")
        else:
            print("[DEBUG VALIDATOR] All documents rejected.")

        return any_relevant
