import csv
from pathlib import Path


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load document texts and labels from a CSV dataset."""
    texts = []
    labels = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])

    return texts, labels