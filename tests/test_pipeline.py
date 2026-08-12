from pathlib import Path

from src.services.document_processor import process_document


def test_process_document_pipeline(tmp_path: Path):
    file = tmp_path / "example.txt"

    file.write_text(
        "  Artificial   Intelligence\n\n"
        "is important.  ",
        encoding="utf-8",
    )

    document = process_document(file)

    assert document.content == (
        "Artificial Intelligence is important."
    )

    assert document.filename == "example.txt"
    assert document.extension == ".txt"