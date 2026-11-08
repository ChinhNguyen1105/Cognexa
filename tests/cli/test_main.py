from unittest.mock import patch

import pytest

from cognexa.cli.main import main


class TestCLIMain:

    @patch("cognexa.cli.main.chat")
    def test_assistant_chat(self, mock_chat):
        with patch(
            "sys.argv",
            ["cognexa", "assistant", "chat"],
        ):
            main()

        mock_chat.assert_called_once()

    @patch("cognexa.cli.main.add_document")
    def test_knowledge_add(self, mock_add_document):
        with patch(
            "sys.argv",
            ["cognexa", "knowledge", "add", "document.txt"],
        ):
            main()

        mock_add_document.assert_called_once_with(
            "document.txt"
        )

    def test_main_without_command_prints_help(self, capsys):
        with patch("sys.argv", ["cognexa"]):
            main()

        captured = capsys.readouterr()

        assert "usage:" in captured.out
        assert "assistant" in captured.out
        assert "knowledge" in captured.out

    def test_assistant_without_subcommand_prints_root_help(self, capsys):
        with patch(
            "sys.argv",
            ["cognexa", "assistant"],
        ):
            main()

        captured = capsys.readouterr()

        assert "usage:" in captured.out
        assert "assistant" in captured.out
        assert "knowledge" in captured.out

    def test_knowledge_add_without_file_path(self):
        with patch(
            "sys.argv",
            ["cognexa", "knowledge", "add"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 2

    def test_help_argument(self, capsys):
        with patch(
            "sys.argv",
            ["cognexa", "--help"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0

        captured = capsys.readouterr()

        assert "Cognexa AI Platform" in captured.out
