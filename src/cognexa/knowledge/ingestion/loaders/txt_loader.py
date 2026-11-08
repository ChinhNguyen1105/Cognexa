from pathlib import Path
from typing import Any

from cognexa.knowledge.ingestion.loaders.base import BaseLoader


class TxtLoader(BaseLoader):
    """
    Loader for plain text documents.

    Supports UTF-8 and UTF-8 with BOM encoding.
    """

    SUPPORTED_ENCODING = ("utf-8-sig", "utf-8")

    def load(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load a TXT document.

        Args:
            file_path: Path to the TXT file.

        Returns:
            A dictionary containing document content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported or
                        the content cannot be decoded.
            IsADirectoryError: If the given path is a directory.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {path}")

        if path.suffix.lower() != ".txt":
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                "TxtLoader only supports .txt files."
            )

        content = self._read_content(path)

        return {
            "content": content,
            "metadata": {
                "source": str(path.resolve()),
                "file_name": path.name,
                "file_type": "txt",
            },
        }

    def _read_content(self, path: Path) -> str:
        """
        Read text content using supported encodings.
        """
        for encoding in self.SUPPORTED_ENCODING:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError(
            f"Unable to decode document: {path}. "
            "Please provide a valid UTF-8 encoded TXT file."
        )
