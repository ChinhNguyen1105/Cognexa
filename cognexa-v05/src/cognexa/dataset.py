import csv

import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    """Dataset for Transformer document classification."""

    def __init__(self, path: str):
        self.samples = []

        with open(path, encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                self.samples.append(
                    (
                        row["text"],
                        int(row["label"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        text, label = self.samples[index]

        return {
            "text": text,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }