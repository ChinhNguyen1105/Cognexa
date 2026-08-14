from pathlib import Path


def load_text_dataset(path: Path) -> list[tuple[str, str]]:
    """Load a labeled text dataset.

    Expected format:
        label<TAB>text
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            label, text = line.split("\t", maxsplit=1)
            samples.append((label, text))

    return samples