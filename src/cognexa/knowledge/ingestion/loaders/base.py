from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLoader(ABC):
    """
    Base class for all document loaders.

    Every concrete loader must implement the load() method
    and return document content with its metadata.
    """

    @abstractmethod
    def load(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load a document from the given file path.

        Args:
            file_path: Path to the source document.

        Returns:
            A dictionary containing:
                - content: Extracted text.
                - metadata: Information about the source document.

        Raises:
            FileNotFoundError: If the document does not exist.
            ValueError: If the document cannot be processed.
        """
        raise NotImplementedError
