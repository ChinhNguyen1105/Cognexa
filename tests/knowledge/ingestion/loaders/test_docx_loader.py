import pytest
from docx import Document

from cognexa.knowledge.ingestion.loaders.docx_loader import DocxLoader


class TestDocxLoader:

    def setup_method(self):
        self.loader = DocxLoader()

    def test_load_valid_docx_file(self, tmp_path):
        # Arrange
        file_path = tmp_path / "sample.docx"

        document = Document()
        document.add_paragraph("Cognexa AI Project")
        document.add_paragraph("Document ingestion module.")

        document.save(file_path)

        # Act
        result = self.loader.load(file_path)

        # Assert
        assert "Cognexa AI Project" in result["content"]
        assert "Document ingestion module." in result["content"]

    def test_load_extracts_table_content(self, tmp_path):
        # Arrange
        file_path = tmp_path / "table.docx"

        document = Document()
        document.add_paragraph("Student information")

        table = document.add_table(rows=2, cols=2)

        table.cell(0, 0).text = "Name"
        table.cell(0, 1).text = "Age"
        table.cell(1, 0).text = "John"
        table.cell(1, 1).text = "20"

        document.save(file_path)

        # Act
        result = self.loader.load(file_path)

        # Assert
        content = result["content"]

        assert "[Table 1]" in content
        assert "Name | Age" in content
        assert "John | 20" in content

    def test_load_returns_correct_metadata(self, tmp_path):
        # Arrange
        file_path = tmp_path / "metadata.docx"

        document = Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("Second paragraph")

        document.add_table(rows=2, cols=2)

        document.save(file_path)

        # Act
        result = self.loader.load(file_path)

        # Assert
        metadata = result["metadata"]

        assert metadata["source"] == str(file_path.resolve())
        assert metadata["file_name"] == "metadata.docx"
        assert metadata["file_type"] == "docx"
        assert metadata["paragraph_count"] == 2
        assert metadata["table_count"] == 1

    def test_load_skips_empty_paragraphs(self, tmp_path):
        # Arrange
        file_path = tmp_path / "empty_paragraph.docx"

        document = Document()
        document.add_paragraph("First paragraph")
        document.add_paragraph("")
        document.add_paragraph("Last paragraph")

        document.save(file_path)

        # Act
        result = self.loader.load(file_path)

        # Assert
        content = result["content"]

        assert "First paragraph" in content
        assert "Last paragraph" in content

        # Empty paragraphs are not included in extracted content.
        assert content == "First paragraph\n\nLast paragraph"

        # Metadata still counts all paragraphs.
        assert result["metadata"]["paragraph_count"] == 3

    def test_load_nonexistent_file(self, tmp_path):
        # Arrange
        file_path = tmp_path / "missing.docx"

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
        with pytest.raises(ValueError, match="only supports .docx"):
            self.loader.load(file_path)

    def test_load_corrupted_docx_raises_error(self, tmp_path):
        # Arrange
        file_path = tmp_path / "corrupted.docx"
        file_path.write_bytes(b"invalid docx content")

        # Act & Assert
        with pytest.raises(ValueError, match="Failed to read DOCX document"):
            self.loader.load(file_path)


