import pytest

from cognexa.knowledge.ingestion.loaders.pdf_loader import PdfLoader


class FakePage:
    def __init__(self, content):
        self.content = content

    def extract_text(self):
        return self.content


class FakePdfReader:
    def __init__(self, pages, is_encrypted=False, decrypt_result=1):
        self.pages = pages
        self.is_encrypted = is_encrypted
        self.decrypt_result = decrypt_result

    def decrypt(self, password):
        return self.decrypt_result


class TestPdfLoader:

    def setup_method(self):
        self.loader = PdfLoader()

    def test_load_valid_pdf_file(self, tmp_path, monkeypatch):
        # Arrange
        file_path = tmp_path / "sample.pdf"
        file_path.write_bytes(b"fake pdf content")

        fake_pages = [
            FakePage("First page content."),
            FakePage("Second page content."),
        ]

        monkeypatch.setattr(
            "cognexa.knowledge.ingestion.loaders.pdf_loader.PdfReader",
            lambda path: FakePdfReader(fake_pages)
        )

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert result["content"] == (
            "First page content.\n\nSecond page content."
        )

    def test_load_returns_correct_metadata(self, tmp_path, monkeypatch):
        # Arrange
        file_path = tmp_path / "sample.pdf"
        file_path.write_bytes(b"fake pdf content")

        fake_pages = [
            FakePage("Page 1"),
            FakePage("Page 2"),
        ]

        monkeypatch.setattr(
            "cognexa.knowledge.ingestion.loaders.pdf_loader.PdfReader",
            lambda path: FakePdfReader(fake_pages)
        )

        # Act
        result = self.loader.load(file_path)

        # Assert
        metadata = result["metadata"]

        assert metadata["source"] == str(file_path.resolve())
        assert metadata["file_name"] == "sample.pdf"
        assert metadata["file_type"] == "pdf"
        assert metadata["page_count"] == 2
        assert metadata["pages"] == [
            {
                "page_number": 1,
                "content": "Page 1",
            },
            {
                "page_number": 2,
                "content": "Page 2",
            },
        ]

    def test_load_skips_empty_pages_when_joining_content(
        self,
        tmp_path,
        monkeypatch
    ):
        # Arrange
        file_path = tmp_path / "sample.pdf"
        file_path.write_bytes(b"fake pdf content")

        fake_pages = [
            FakePage("First page"),
            FakePage(""),
            FakePage("   "),
            FakePage("Last page"),
        ]

        monkeypatch.setattr(
            "cognexa.knowledge.ingestion.loaders.pdf_loader.PdfReader",
            lambda path: FakePdfReader(fake_pages)
        )

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert result["content"] == "First page\n\nLast page"

        # Metadata vẫn lưu đầy đủ các trang, kể cả trang rỗng.
        assert result["metadata"]["page_count"] == 4
        assert len(result["metadata"]["pages"]) == 4

    def test_load_nonexistent_file(self, tmp_path):
        # Arrange
        file_path = tmp_path / "missing.pdf"

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

    def test_load_unsupported_extension(self, tmp_path):
        # Arrange
        file_path = tmp_path / "sample.txt"
        file_path.write_text("Some content", encoding="utf-8")

        # Act & Assert
        with pytest.raises(ValueError, match="only supports .pdf"):
            self.loader.load(file_path)

    def test_load_corrupted_pdf_raises_error(self, tmp_path, monkeypatch):
        # Arrange
        file_path = tmp_path / "corrupted.pdf"
        file_path.write_bytes(b"invalid pdf content")

        def raise_error(path):
            raise RuntimeError("Invalid PDF structure")

        monkeypatch.setattr(
            "cognexa.knowledge.ingestion.loaders.pdf_loader.PdfReader",
            raise_error
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Failed to read PDF document"):
            self.loader.load(file_path)

    def test_load_encrypted_pdf_raises_error(self, tmp_path, monkeypatch):
        # Arrange
        file_path = tmp_path / "encrypted.pdf"
        file_path.write_bytes(b"fake encrypted pdf")

        fake_reader = FakePdfReader(
            pages=[],
            is_encrypted=True,
            decrypt_result=0
        )

        monkeypatch.setattr(
            "cognexa.knowledge.ingestion.loaders.pdf_loader.PdfReader",
            lambda path: fake_reader
        )

        # Act & Assert
        with pytest.raises(ValueError, match="PDF document is encrypted"):
            self.loader.load(file_path)


