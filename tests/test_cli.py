from typer.testing import CliRunner

from cognexa.cli.main import app


runner = CliRunner()


def test_process_document():
    result = runner.invoke(
        app,
        ["process", "data/samples/documents/example.txt"],
    )

    assert result.exit_code == 0
    assert "Artificial intelligence" in result.stdout

def test_version():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "Cognexa 0.1.0" in result.stdout