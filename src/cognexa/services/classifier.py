from pathlib import Path

from cognexa.ml.model import load_model


def classify_text(text: str, model_path: Path) -> str:
    """Classify text using a trained document model."""
    trainer = load_model(model_path)

    predictions = trainer.predict([text])

    return predictions[0]