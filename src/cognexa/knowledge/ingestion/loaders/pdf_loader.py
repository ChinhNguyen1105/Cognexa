from pathlib import Path
from typing import Any

from pypdf import PdfReader

from cognexa.knowledge.ingestion.loaders.base import BaseLoader


class PdfLoader(BaseLoader):
    """
    Loader for PDF documents.

    Extracts text from each page and preserves page-level metadata.
    """

    def load(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A dictionary containing:
                - content: Combined text from all pages.
                - metadata: Document information and page-level content.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the given path is a directory.
            ValueError: If the file extension is unsupported,
                        the PDF is encrypted, or cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.is_dir():
            raise IsADirectoryError(
                f"Expected a file, got a directory: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                "PdfLoader only supports .pdf files."
            )

        pages = self._extract_pages(path)

        content = "\n\n".join(
            page["content"]
            for page in pages
            if page["content"].strip()
        )

        return {
            "content": content,
            "metadata": {
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "pdf",
                "page_count": len(pages),
                "pages": pages,
            },
        }

    def _extract_pages(self, path: Path) -> list[dict[str, Any]]:
        """
        Extract text from every page in a PDF document.
        """
        try:
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                decrypt_result = reader.decrypt("")

                if decrypt_result == 0:
                    raise ValueError(
                        f"PDF document is encrypted: {path}"
                    )

            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                page_content = page.extract_text() or ""

                pages.append({
                    "page_number": page_number,
                    "content": page_content,
                })

            return pages

        except ValueError:
            raise

        except Exception as error:
            raise ValueError(
                f"Failed to read PDF document: {path}. "
                f"Reason: {error}"
            ) from error


