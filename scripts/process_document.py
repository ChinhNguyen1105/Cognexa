import sys
from pathlib import Path

from src.core.exceptions import CognexaError
from src.services.document_processor import process_document


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m scripts.process_document <file>"
        )
        return 1

    file_path = Path(sys.argv[1])

    print("=== Cognexa Document Processor ===")
    print(f"Source: {file_path}")

    try:
        document = process_document(file_path)

    except CognexaError as exc:
        print(f"\nError: {exc}")
        return 1

    print("\n=== Processed Content ===")
    print(document.content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())