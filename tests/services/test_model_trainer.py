from pathlib import Path

from cognexa.ml.model import load_model
from cognexa.services.model_trainer import train_model


def test_train_model(tmp_path: Path):
    dataset_path = Path("data/samples/datasets/documents.csv")
    model_path = tmp_path / "cognexa_model.pkl"

    train_model(dataset_path, model_path)

    assert model_path.exists()

    trainer = load_model(model_path)

    predictions = trainer.predict(
        ["machine learning neural networks"]
    )

    assert predictions[0] == "ai"