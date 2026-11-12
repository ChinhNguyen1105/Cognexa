from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._validation import (
    require_non_negative_int,
    require_text,
)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    parent_chunk_id: str | None = None
    position: int | None = None

    def __post_init__(self) -> None:
        require_text(self.chunk_id, "chunk_id")
        require_text(self.document_id, "document_id")
        require_text(self.content, "content")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

        if self.parent_chunk_id is not None:
            require_text(self.parent_chunk_id, "parent_chunk_id")

        if self.position is not None:
            require_non_negative_int(self.position, "position")