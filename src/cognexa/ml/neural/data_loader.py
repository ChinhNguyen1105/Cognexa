import csv
from pathlib import Path


def load_neural_dataset(
    path: str | Path,
) -> tuple[list[str], list[int]]:
    """Load document texts and labels from a CSV dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    texts = []
    labels = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))

    if not texts:
        raise ValueError("Dataset is empty.")

    return texts, labels