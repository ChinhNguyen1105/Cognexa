import pickle
from pathlib import Path


def save_model(model, path: Path) -> None:
    """Save a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(model, file)


def load_model(path: Path):
    """Load a trained model from disk."""
    with path.open("rb") as file:
        return pickle.load(file)