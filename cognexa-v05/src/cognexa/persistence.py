import torch


def save_model(model, path: str) -> None:
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)


def load_model(model, path: str) -> None:
    """Load model weights from disk."""
    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)