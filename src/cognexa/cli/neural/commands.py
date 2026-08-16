import torch
import typer

from cognexa.services.neural_classifier import NeuralDocumentClassifier

app = typer.Typer(
    name="neural",
    help="Neural document classification commands.",
)


@app.command()
def train(
    epochs: int = typer.Option(10, "--epochs"),
):
    """Train a neural document classifier."""

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    classifier = NeuralDocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    history = classifier.train(
        features=features,
        labels=labels,
        epochs=epochs,
        batch_size=2,
    )

    classifier.save("models/neural/document_classifier.pth")

    typer.echo(
        f"Training completed. Final loss: {history[-1]:.4f}"
    )


@app.command()
def predict():
    """Run neural document prediction."""

    classifier = NeuralDocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    classifier.load(
        "models/neural/document_classifier.pth"
    )

    features = torch.tensor([
        [1.0, 0.0],
    ])

    predictions = classifier.predict(features)

    typer.echo(
        f"Prediction: {predictions.tolist()}"
    )


@app.command()
def evaluate():
    """Evaluate the trained neural classifier."""

    classifier = NeuralDocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    classifier.load(
        "models/neural/document_classifier.pth"
    )

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = torch.tensor([0, 1, 0, 1])

    accuracy = classifier.evaluate(
        features,
        labels,
    )

    typer.echo(
        f"Accuracy: {accuracy:.2%}"
    )
