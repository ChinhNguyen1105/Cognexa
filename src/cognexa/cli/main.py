from pathlib import Path

import typer

from cognexa.core.exceptions import DocumentNotFoundError
from cognexa.services.document_processor import process_document

app = typer.Typer(
    name="cognexa",
    help="Cognexa - Document Processing Intelligence",
)


@app.command()
def process(file: str):
    """Process a document."""
    try:
        document = process_document(Path(file))
        print(document.content)

    except DocumentNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def version():
    """Show Cognexa version."""
    print("Cognexa 0.1.0")


if __name__ == "__main__":
    app()