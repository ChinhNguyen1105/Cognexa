from pathlib import Path

from cognexa.services.classifier import classify_text


def test_classify_text(tmp_path: Path):
    model_path = tmp_path / "model.pkl"

    from cognexa.ml.model import save_model
    from cognexa.ml.classical.trainer import DocumentTrainer

    trainer = DocumentTrainer()

    trainer.train(
        [
            "Python programming code",
            "Python software development",
            "football match player",
            "football team tournament",
        ],
        [
            "technology",
            "technology",
            "sports",
            "sports",
        ],
    )

    save_model(trainer, model_path)

    result = classify_text(
        "Python programming language",
        model_path,
    )

    assert result == "technology"