from typer.testing import CliRunner

from cognexa.cli.main import app

runner = CliRunner()


def test_process_document_not_found():
    result = runner.invoke(
        app,
        ["process", "does_not_exist.txt"],
    )

    assert result.exit_code != 0