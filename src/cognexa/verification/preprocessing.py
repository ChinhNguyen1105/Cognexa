import re


class TextPreprocessor:
    CONTEXT_HEADER_PREFIX = "[Document "

    LIST_ITEM_PATTERN = re.compile(
        r"^\s*(?:[-*•‣▪◦]|\d+[.)])\s+(.+?)\s*$"
    )

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        text = text.casefold()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)

        return text.strip()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        text = text.strip()

        if not text:
            return []

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @classmethod
    def is_context_header(cls, line: str) -> bool:
        """True for document headers added by context builders."""
        return line.strip().startswith(cls.CONTEXT_HEADER_PREFIX)

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        text = text.strip()

        if text and text[-1] not in ".!?":
            return f"{text}."

        return text

    @classmethod
    def _extract_list_item(cls, line: str) -> str | None:
        match = cls.LIST_ITEM_PATTERN.match(line)

        if not match:
            return None

        return match.group(1).strip()

    @classmethod
    def split_evidence(cls, text: str) -> list[str]:
        """
        Split retrieved context into standalone evidence units.

        Supports:
        - Document header removal.
        - Sentence and line splitting.
        - Bulleted and numbered lists.
        - Carrying a colon-terminated lead-in into each list item.
        - Duplicate removal.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        if not text.strip():
            return []

        units = []
        seen = set()
        pending_lead_in = None
        lead_in_consumed = False

        def add_units(content: str):
            for sentence in cls.split_sentences(content):
                sentence = " ".join(sentence.split())

                if not sentence:
                    continue

                normalized = cls.normalize(sentence)

                if normalized and normalized not in seen:
                    seen.add(normalized)
                    units.append(sentence)

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line or cls.is_context_header(line):
                continue

            list_item = cls._extract_list_item(line)

            if list_item is not None:
                if pending_lead_in:
                    combined = (
                        f"{pending_lead_in} "
                        f"{cls._ensure_terminal_punctuation(list_item)}"
                    )
                    add_units(combined)
                    lead_in_consumed = True
                else:
                    add_units(
                        cls._ensure_terminal_punctuation(list_item)
                    )

                continue

            if pending_lead_in:
                if not lead_in_consumed:
                    add_units(pending_lead_in)

                pending_lead_in = None
                lead_in_consumed = False

            if line.endswith(":"):
                pending_lead_in = line
                continue

            add_units(line)

        if pending_lead_in and not lead_in_consumed:
            add_units(pending_lead_in)

        return units
    @classmethod
    def preprocess(cls, text: str) -> list[str]:
        return [
            cls.normalize(sentence)
            for sentence in cls.split_sentences(text)
            if cls.normalize(sentence)
        ]

