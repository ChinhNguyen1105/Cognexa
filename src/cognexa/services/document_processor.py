from pathlib import Path

from cognexa.documents.models import Document
from cognexa.ingestion.loader import load_document
from cognexa.processing.processor import process_document_content


def process_document(path: Path) -> Document:
    document = load_document(path)

    processed_content = process_document_content(document)

    return Document(
        content=processed_content,
        source=document.source,
        metadata=document.metadata,
    )