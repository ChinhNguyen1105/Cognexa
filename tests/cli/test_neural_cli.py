from typer.testing import CliRunner

from cognexa.cli.main import app


runner = CliRunner()


def test_neural_train(tmp_path):
    model_path = (
        tmp_path
        / "document_classifier.pth"
    )

    result = runner.invoke(
        app,
        [
            "neural",
            "train",
            "--epochs",
            "20",
            "--batch-size",
            "4",
            "--model",
            str(model_path),
        ],
    )

    assert result.exit_code == 0

    assert model_path.exists()

    assert "Training completed" in (
        result.stdout
    )


def test_neural_predict(tmp_path):
    model_path = (
        tmp_path
        / "document_classifier.pth"
    )

    train_result = runner.invoke(
        app,
        [
            "neural",
            "train",
            "--epochs",
            "50",
            "--model",
            str(model_path),
        ],
    )

    assert train_result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "neural",
            "predict",
            "data/samples/documents/example.txt",
            "--model",
            str(model_path),
        ],
    )

    assert result.exit_code == 0
    assert "Prediction:" in result.stdout