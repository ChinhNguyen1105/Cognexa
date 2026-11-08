import pytest

from cognexa.knowledge.ingestion.loader import DocumentLoader


class FakeLoader:

    def __init__(self):
        self.called = False
        self.received_path = None

    def load(self, file_path):
        self.called = True
        self.received_path = file_path

        return {
            "content": "Fake document content",
            "metadata": {
                "file_type": file_path.suffix.lower().replace(".", "")
            },
        }


class TestDocumentLoader:

    def setup_method(self):
        self.loader = DocumentLoader()

    @pytest.mark.parametrize(
        ("extension", "expected_loader"),
        [
            (".txt", "txt"),
            (".pdf", "pdf"),
            (".docx", "docx"),
        ],
    )
    def test_selects_correct_loader(
        self,
        tmp_path,
        monkeypatch,
        extension,
        expected_loader,
    ):
        # Arrange
        file_path = tmp_path / f"sample{extension}"
        file_path.write_bytes(b"fake document")

        fake_loader = FakeLoader()
        loader_called = {"value": False}

        def fake_loader_factory():
            loader_called["value"] = True
            return fake_loader

        monkeypatch.setitem(
            DocumentLoader.SUPPORTED_LOADERS,
            extension,
            lambda: fake_loader_factory(),
        )

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert loader_called["value"] is True
        assert fake_loader.called is True
        assert fake_loader.received_path == file_path
        assert result["metadata"]["file_type"] == expected_loader

    def test_load_unsupported_format(self, tmp_path):
        # Arrange
        file_path = tmp_path / "sample.xlsx"
        file_path.write_bytes(b"fake document")

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Supported formats",
        ):
            self.loader.load(file_path)

    def test_load_nonexistent_file(self, tmp_path):
        # Arrange
        file_path = tmp_path / "missing.txt"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            self.loader.load(file_path)

    def test_load_directory_raises_error(self, tmp_path):
        # Arrange
        directory_path = tmp_path / "documents"
        directory_path.mkdir()

        # Act & Assert
        with pytest.raises(IsADirectoryError):
            self.loader.load(directory_path)

    def test_load_uses_case_insensitive_extension(
        self,
        tmp_path,
        monkeypatch,
    ):
        # Arrange
        file_path = tmp_path / "sample.TXT"
        file_path.write_bytes(b"fake document")

        fake_loader = FakeLoader()

        monkeypatch.setitem(
            DocumentLoader.SUPPORTED_LOADERS,
            ".txt",
            lambda: fake_loader,
        )

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert fake_loader.called is True
        assert result["metadata"]["file_type"] == "txt"
