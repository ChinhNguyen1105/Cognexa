from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    content: str
    source: Path
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.source.name

    @property
    def extension(self) -> str:
        return self.source.suffix.lower()