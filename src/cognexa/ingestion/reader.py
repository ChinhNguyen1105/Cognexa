from pathlib import Path

from cognexa.core.config import DEFAULT_ENCODING
from cognexa.core.exceptions import DocumentReadError


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding=DEFAULT_ENCODING)
    except (OSError, UnicodeError) as exc:
        raise DocumentReadError(
            f"Unable to read document: {path}"
        ) from exc