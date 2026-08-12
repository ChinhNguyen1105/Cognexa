from pathlib import Path

from src.core.config import SUPPORTED_EXTENSIONS
from src.core.exceptions import (
    DocumentNotFoundError,
    UnsupportedDocumentError,
)
from src.documents.models import Document
from src.ingestion.reader import read_text_file


def load_document(path: Path) -> Document:
    path = Path(path)

    if not path.exists():
        raise DocumentNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():
        raise DocumentNotFoundError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type: {path.suffix}"
        )

    content = read_text_file(path)

    metadata = {
        "filename": path.name,
        "extension": path.suffix.lower(),
    }

    return Document(
        content=content,
        source=path,
        metadata=metadata,
    )