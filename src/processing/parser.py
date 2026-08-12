from src.documents.models import Document


def parse(document: Document) -> str:
    return document.content