import re


class TextChunker:
    """
    Split long text into smaller chunks.

    Prioritizes sentence boundaries, falls back to whitespace
    boundaries, and preserves word integrity when applying overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _find_chunk_end(
        self,
        text: str,
        start: int,
        target_end: int,
    ) -> int:
        """Find a suitable chunk boundary."""

        min_boundary = start + self.chunk_size // 2

        sentence_boundaries = [
            start + match.end()
            for match in re.finditer(
                r"[.!?](?=\s|$)",
                text[start:target_end],
            )
        ]

        sentence_boundaries = [
            position
            for position in sentence_boundaries
            if position >= min_boundary
        ]

        if sentence_boundaries:
            return sentence_boundaries[-1]

        whitespace_position = max(
            text.rfind(" ", min_boundary, target_end),
            text.rfind("\n", min_boundary, target_end),
            text.rfind("\t", min_boundary, target_end),
        )

        if whitespace_position > start:
            return whitespace_position

        return target_end

    def _find_next_start(
        self,
        text: str,
        current_end: int,
    ) -> int:
        """Calculate the next chunk start."""

        next_start = max(
            0,
            current_end - self.chunk_overlap,
        )

        while (
            next_start < len(text)
            and next_start > 0
            and not text[next_start - 1].isspace()
            and not text[next_start].isspace()
        ):
            next_start += 1

        while (
            next_start < len(text)
            and text[next_start].isspace()
        ):
            next_start += 1

        return next_start

    def chunk_text(self, text: str) -> list[str]:
        """Split source text into overlapping chunks."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            target_end = min(
                start + self.chunk_size,
                text_length,
            )

            if target_end >= text_length:
                chunk = text[start:].strip()

                if chunk:
                    chunks.append(chunk)

                break

            end = self._find_chunk_end(
                text=text,
                start=start,
                target_end=target_end,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            next_start = self._find_next_start(
                text=text,
                current_end=end,
            )

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def chunk_document(
        self,
        text: str,
        child_chunker: "TextChunker",
    ) -> list[dict]:
        """
        Split a document into parent chunks and child chunks.

        Each child chunk belongs to exactly one parent.
        """

        parent_chunks = self.chunk_text(text)

        result = []

        for parent_index, parent_content in enumerate(parent_chunks):
            child_chunks = child_chunker.chunk_text(parent_content)

            result.append(
                {
                    "parent_index": parent_index,
                    "content": parent_content,
                    "children": [
                        {
                            "chunk_index": child_index,
                            "content": child_content,
                        }
                        for child_index, child_content
                        in enumerate(child_chunks)
                    ],
                }
            )

        return result