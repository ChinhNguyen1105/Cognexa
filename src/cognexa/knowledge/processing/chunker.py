
import re


class TextChunker:
    """
    Split long text into smaller chunks.

    Prioritizes sentence boundaries, falls back to whitespace
    boundaries, and preserves word integrity when applying overlap.
    """

    LIST_ITEM_PATTERN = re.compile(
        r"^\s*(?:[-*•‣▪◦]|\d+[.)])\s+(.+?)\s*$"
    )

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
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _find_chunk_end(self, text, start, target_end):
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

    def _find_next_start(self, text, current_end):
        next_start = max(0, current_end - self.chunk_overlap)

        while (
            next_start < len(text)
            and next_start > 0
            and not text[next_start - 1].isspace()
            and not text[next_start].isspace()
        ):
            next_start += 1

        while next_start < len(text) and text[next_start].isspace():
            next_start += 1

        return next_start

    def _chunk_text_with_spans(self, text: str) -> list[dict]:
        """
        Split text and preserve each chunk's character offsets.

        Offsets are relative to the stripped input text.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [{
                "content": text,
                "start": 0,
                "end": len(text),
            }]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            target_end = min(
                start + self.chunk_size,
                text_length,
            )

            if target_end >= text_length:
                raw_chunk = text[start:text_length]

                left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
                right_trimmed = raw_chunk.rstrip()

                chunk_start = start + left_trim
                chunk_end = start + len(right_trimmed)
                chunk_content = text[chunk_start:chunk_end]

                if chunk_content:
                    chunks.append({
                        "content": chunk_content,
                        "start": chunk_start,
                        "end": chunk_end,
                    })

                break

            end = self._find_chunk_end(
                text,
                start,
                target_end,
            )

            raw_chunk = text[start:end]

            left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            right_trimmed = raw_chunk.rstrip()

            chunk_start = start + left_trim
            chunk_end = start + len(right_trimmed)
            chunk_content = text[chunk_start:chunk_end]

            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "start": chunk_start,
                    "end": chunk_end,
                })

            next_start = self._find_next_start(text, end)

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Public API remains unchanged.
        """
        return [
            chunk["content"]
            for chunk in self._chunk_text_with_spans(text)
        ]

    def _extract_list_groups(self, text: str) -> list[dict]:
        """
        Detect consecutive list items in the original text.

        Each group represents one independent list.
        Item indices are local to their respective groups.
        """
        groups = []

        current_group = None
        current_offset = 0

        for line in text.splitlines(keepends=True):
            line_content = line.rstrip("\r\n")

            match = self.LIST_ITEM_PATTERN.match(line_content)

            if match:
                if current_group is None:
                    current_group = {
                        "group_id": len(groups),
                        "items": [],
                    }

                current_group["items"].append({
                    "item_index": len(current_group["items"]),
                    "start": current_offset,
                    "end": current_offset + len(line_content),
                })

            else:
                if current_group is not None:
                    groups.append(current_group)
                    current_group = None

            current_offset += len(line)

        if current_group is not None:
            groups.append(current_group)

        return groups

    def _map_list_groups_to_child(
        self,
        child_start: int,
        child_end: int,
        list_groups: list[dict],
    ) -> list[dict]:
        """
        Map list items intersecting a child chunk to its metadata.
        """
        child_groups = []

        for group in list_groups:
            item_indices = []

            for item in group["items"]:
                overlaps = (
                    child_start < item["end"]
                    and child_end > item["start"]
                )

                if overlaps:
                    item_indices.append(item["item_index"])

            if item_indices:
                child_groups.append({
                    "group_id": group["group_id"],
                    "item_indices": item_indices,
                })

        return child_groups

    def chunk_document(
        self,
        text: str,
        child_chunker: "TextChunker",
    ) -> list[dict]:
        """
        Split a document into parent chunks and child chunks.

        Each child belongs to exactly one parent.
        List metadata identifies the list groups and items
        intersecting each child chunk.
        """
        parent_chunks = self.chunk_text(text)

        result = []

        for parent_index, parent_content in enumerate(parent_chunks):
            child_texts = child_chunker.chunk_text(parent_content)

            child_chunks = []
            search_from = 0

            for child_content in child_texts:
                start = parent_content.find(child_content, search_from)

                if start < 0:
                    start = parent_content.find(child_content)

                if start < 0:
                    start = 0
                    end = 0
                else:
                    end = start + len(child_content)
                    search_from = start + 1

                child_chunks.append({
                    "content": child_content,
                    "start": start,
                    "end": end,
                })

            list_groups = self._extract_list_groups(parent_content)

            result.append({
                "parent_index": parent_index,
                "content": parent_content,
                "children": [
                    {
                        "chunk_index": child_index,
                        "content": child["content"],
                        "list_groups": self._map_list_groups_to_child(
                            child_start=child["start"],
                            child_end=child["end"],
                            list_groups=list_groups,
                        ),
                    }
                    for child_index, child in enumerate(child_chunks)
                ],
            })

        return result


