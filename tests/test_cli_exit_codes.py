from typer.testing import CliRunner

from cognexa.cli.main import app

runner = CliRunner()


def test_process_success_exit_code():
    result = runner.invoke(
        app,
        ["process", "data/samples/documents/example.txt"],
    )

    assert result.exit_code == 0


def test_process_missing_file_exit_code():
    result = runner.invoke(
        app,
        ["process", "does_not_exist.txt"],
    )

    assert result.exit_code == 1


def test_version_exit_code():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0