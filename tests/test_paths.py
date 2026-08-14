from pathlib import Path


def test_sample_document_path_is_platform_independent():
    path = (
        Path("data")
        / "samples"
        / "documents"
        / "example.txt"
    )

    assert path.name == "example.txt"
    assert path.suffix == ".txt"
    assert path.parent.name == "documents"