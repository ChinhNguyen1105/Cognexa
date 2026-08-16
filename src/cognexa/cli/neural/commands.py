from pathlib import Path

import typer

from cognexa.ml.dataset import load_dataset
from cognexa.services.document_processor import process_document
from cognexa.services.neural_classifier import (
    NeuralDocumentClassifier,
)

app = typer.Typer(
    name="neural",
    help="Neural document classification commands.",
)

DEFAULT_DATASET = (
    "data/samples/datasets/documents.csv"
)

DEFAULT_MODEL = (
    "models/neural/document_classifier.pth"
)


@app.command()
def train(
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
    ),
    hidden_size: int = typer.Option(
        32,
        "--hidden-size",
    ),
    learning_rate: float = typer.Option(
        0.001,
        "--learning-rate",
    ),
):
    """Train the neural document classifier."""

    texts, labels = load_dataset(
        Path(dataset)
    )

    classifier = NeuralDocumentClassifier(
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )

    history = classifier.train(
        texts=texts,
        labels=labels,
        epochs=epochs,
        batch_size=batch_size,
    )

    classifier.save(model)

    typer.echo(
        f"Training completed."
    )

    typer.echo(
        f"Initial loss: {history[0]:.4f}"
    )

    typer.echo(
        f"Final loss: {history[-1]:.4f}"
    )

    typer.echo(
        f"Model saved to: {model}"
    )


@app.command()
def evaluate(
    dataset: str = typer.Option(
        DEFAULT_DATASET,
        "--dataset",
        "-d",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
    ),
):
    """Evaluate the trained neural classifier."""

    texts, labels = load_dataset(
        Path(dataset)
    )

    classifier = NeuralDocumentClassifier()

    classifier.load(model)

    accuracy = classifier.evaluate(
        texts=texts,
        labels=labels,
    )

    typer.echo(
        f"Accuracy: {accuracy:.2%}"
    )


@app.command()
def predict(
    file: str = typer.Argument(
        ...,
        help="Document to classify.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
    ),
):
    """Classify a document using the neural model."""

    document = process_document(
        Path(file)
    )

    classifier = NeuralDocumentClassifier()

    classifier.load(model)

    prediction = classifier.predict(
        [document.content]
    )[0]

    typer.echo(
        f"Document: {document.filename}"
    )

    typer.echo(
        f"Prediction: {prediction}"
    )


@app.command()
def info(
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
    ),
):
    """Show information about a trained neural model."""

    classifier = NeuralDocumentClassifier()

    classifier.load(model)

    typer.echo(
        "=== Cognexa Neural Model ==="
    )

    typer.echo(
        f"Input size: {classifier.input_size}"
    )

    typer.echo(
        f"Hidden size: {classifier.hidden_size}"
    )

    typer.echo(
        f"Classes: {classifier.num_classes}"
    )

    typer.echo(
        f"Labels: "
        f"{list(classifier.label_to_id.keys())}"
    )