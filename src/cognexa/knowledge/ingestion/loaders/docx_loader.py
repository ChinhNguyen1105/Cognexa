from pathlib import Path
from typing import Any

from docx import Document

from cognexa.knowledge.ingestion.loaders.base import BaseLoader


class DocxLoader(BaseLoader):
    """
    Loader for Microsoft Word documents (.docx).

    Extracts text from paragraphs and tables.
    """

    def load(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load a DOCX document.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            A dictionary containing:
                - content: Extracted document text.
                - metadata: Information about the source document.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the given path is a directory.
            ValueError: If the file extension is unsupported
                        or the document cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.is_dir():
            raise IsADirectoryError(
                f"Expected a file, got a directory: {path}"
            )

        if path.suffix.lower() != ".docx":
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                "DocxLoader only supports .docx files."
            )

        paragraphs, tables = self._extract_content(path)

        content_parts = []

        content_parts.extend(
            paragraph
            for paragraph in paragraphs
            if paragraph.strip()
        )

        for table_index, table_rows in enumerate(tables, start=1):
            content_parts.append(f"[Table {table_index}]")

            for row in table_rows:
                content_parts.append(" | ".join(row))

        content = "\n\n".join(content_parts)

        return {
            "content": content,
            "metadata": {
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "docx",
                "paragraph_count": len(paragraphs),
                "table_count": len(tables),
            },
        }

    def _extract_content(
        self,
        path: Path,
    ) -> tuple[list[str], list[list[list[str]]]]:
        """
        Extract paragraphs and table contents from a DOCX document.
        """
        try:
            document = Document(str(path))

            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
            ]

            tables = []

            for table in document.tables:
                table_rows = []

                for row in table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    table_rows.append(cells)

                tables.append(table_rows)

            return paragraphs, tables

        except Exception as error:
            raise ValueError(
                f"Failed to read DOCX document: {path}. "
                f"Reason: {error}"
            ) from error


