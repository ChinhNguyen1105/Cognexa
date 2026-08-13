from pathlib import Path

import pytest

from cognexa.core.exceptions import DocumentNotFoundError
from cognexa.ingestion.loader import load_document


def test_load_document(tmp_path: Path):
    file = tmp_path / "example.txt"
    file.write_text("Hello Cognexa", encoding="utf-8")

    document = load_document(file)

    assert document.content == "Hello Cognexa"
    assert document.source == file
    assert document.metadata["filename"] == "example.txt"


def test_load_document_not_found(tmp_path: Path):
    file = tmp_path / "missing.txt"

    with pytest.raises(DocumentNotFoundError):
        load_document(file)