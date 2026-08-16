
from pathlib import Path
import csv


def load_text_dataset(path: Path) -> list[tuple[str, str]]:
    """Load a labeled text dataset from a CSV file.

    Expected format:
        text,label
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["text", "label"]:
            raise ValueError(
                "Dataset must contain 'text' and 'label' columns."
            )

        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip()

            if not text or not label:
                continue

            samples.append((text, label))

    return samples
