from src.processing.cleaner import clean_text
from src.processing.parser import parse


def process(text: str) -> str:
    return clean_text(text)


def process_document_content(document) -> str:
    parsed = parse(document)
    return clean_text(parsed)