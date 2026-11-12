import pytest

from cognexa.knowledge.ingestion.loaders.txt_loader import TxtLoader


class TestTxtLoader:

    def setup_method(self):
        self.loader = TxtLoader()

    def test_load_valid_txt_file(self, tmp_path):
        # Arrange
        file_path = tmp_path / "sample.txt"
        expected_content = "Hello Cognexa!\nThis is a test document."

        file_path.write_text(expected_content, encoding="utf-8")

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert result["content"] == expected_content

    def test_load_returns_correct_metadata(self, tmp_path):
        # Arrange
        file_path = tmp_path / "sample.txt"
        file_path.write_text("Test content", encoding="utf-8")

        # Act
        result = self.loader.load(file_path)

        # Assert
        metadata = result["metadata"]

        assert metadata["source"] == str(file_path.resolve())
        assert metadata["file_name"] == "sample.txt"
        assert metadata["file_type"] == "txt"

    def test_load_utf8_bom_file(self, tmp_path):
        # Arrange
        file_path = tmp_path / "bom.txt"
        expected_content = "Cognexa supports UTF-8 BOM."

        file_path.write_text(
            expected_content,
            encoding="utf-8-sig"
        )

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert result["content"] == expected_content
        assert not result["content"].startswith("\ufeff")

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

    def test_load_unsupported_extension(self, tmp_path):
        # Arrange
        file_path = tmp_path / "sample.pdf"
        file_path.write_text("Some content", encoding="utf-8")

        # Act & Assert
        with pytest.raises(ValueError, match="only supports .txt"):
            self.loader.load(file_path)

    def test_load_invalid_encoding(self, tmp_path):
        # Arrange
        file_path = tmp_path / "invalid.txt"
        file_path.write_bytes(b"\xff\xfe\xfa")

        # Act & Assert
        with pytest.raises(ValueError, match="Unable to decode document"):
            self.loader.load(file_path)


