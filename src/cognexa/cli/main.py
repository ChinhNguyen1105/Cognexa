from pathlib import Path

import typer

from cognexa.core.config import APP_NAME, APP_VERSION
from cognexa.core.exceptions import DocumentNotFoundError
from cognexa.core.logging import get_logger, setup_logging
from cognexa.services.document_processor import process_document

setup_logging()

logger = get_logger(__name__)

app = typer.Typer(
    name="cognexa",
    help="Cognexa - Document Processing Intelligence",
)


@app.command()
def process(file: str):
    """Process a document."""
    try:
        logger.info("Processing document: %s", file)

        document = process_document(Path(file))
        print(document.content)

    except DocumentNotFoundError as exc:
        logger.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def version():
    """Show Cognexa version."""
    print(f"{APP_NAME} {APP_VERSION}")


if __name__ == "__main__":
    app()