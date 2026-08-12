class CognexaError(Exception):
    """Base exception for Cognexa."""


class DocumentError(CognexaError):
    """Base exception for document-related errors."""


class DocumentNotFoundError(DocumentError):
    """Raised when a document does not exist."""


class UnsupportedDocumentError(DocumentError):
    """Raised when the document format is not supported."""


class DocumentReadError(DocumentError):
    """Raised when a document cannot be read."""