from pathlib import Path

from cognexa.knowledge.ingestion.loaders.base import BaseLoader
from cognexa.knowledge.ingestion.loaders.txt_loader import TxtLoader
from cognexa.knowledge.ingestion.loaders.pdf_loader import PdfLoader
from cognexa.knowledge.ingestion.loaders.docx_loader import DocxLoader


class DocumentLoader:
    """
    Dispatcher for supported document loaders.

    Automatically selects the appropriate loader based
    on the document file extension.
    """

    SUPPORTED_LOADERS: dict[str, type[BaseLoader]] = {
        ".txt": TxtLoader,
        ".pdf": PdfLoader,
        ".docx": DocxLoader,
    }

    def load(self, file_path: str | Path) -> dict:
        """
        Load a document using its corresponding loader.

        Args:
            file_path: Path to the source document.

        Returns:
            Extracted document content and metadata.

        Raises:
            FileNotFoundError: If the document does not exist.
            IsADirectoryError: If the path is a directory.
            ValueError: If the document format is unsupported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.is_dir():
            raise IsADirectoryError(
                f"Expected a file, got a directory: {path}"
            )

        loader_class = self.SUPPORTED_LOADERS.get(path.suffix.lower())

        if loader_class is None:
            supported_formats = ", ".join(self.SUPPORTED_LOADERS)

            raise ValueError(
                f"Unsupported document format: {path.suffix}. "
                f"Supported formats: {supported_formats}"
            )

        loader = loader_class()

        return loader.load(path)
