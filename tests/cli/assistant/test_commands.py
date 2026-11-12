from unittest.mock import MagicMock, call, patch

import pytest

from cognexa.cli.assistant.commands import (
    add_document,
    chat,
)


class TestAddDocument:

    @patch("cognexa.cli.assistant.commands.create_knowledge_base")
    def test_add_document_success(self, mock_create_knowledge_base, capsys):
        knowledge_base = MagicMock()
        knowledge_base.add_document.return_value = {
            "chunk_count": 3,
            "parent_count": 1,
            "child_count": 3,
            "source": "document.txt",
            "file_type": "txt",
        }

        mock_create_knowledge_base.return_value = knowledge_base

        add_document("document.txt")

        mock_create_knowledge_base.assert_called_once_with(
            initialize_sparse_index=False,
        )
        knowledge_base.add_document.assert_called_once_with("document.txt")

        captured = capsys.readouterr()

        assert "Document added successfully." in captured.out
        assert "File: document.txt" in captured.out
        assert "Type: txt" in captured.out
        assert "Parent chunks created: 1" in captured.out
        assert "Child chunks created: 3" in captured.out

    @patch("cognexa.cli.assistant.commands.create_knowledge_base")
    def test_add_document_without_text_content(
        self,
        mock_create_knowledge_base,
        capsys,
    ):
        knowledge_base = MagicMock()
        knowledge_base.add_document.return_value = {
            "chunk_count": 0,
            "parent_count": 0,
            "child_count": 0,
            "source": "empty.pdf",
            "file_type": "pdf",
            "parent_ids": [],
            "child_ids": [],
        }

        mock_create_knowledge_base.return_value = knowledge_base

        add_document("empty.pdf")

        mock_create_knowledge_base.assert_called_once_with(
            initialize_sparse_index=False,
        )
        knowledge_base.add_document.assert_called_once_with("empty.pdf")

        captured = capsys.readouterr()

        assert "No text content found" in captured.out
        assert "Document added successfully." not in captured.out

    @patch("cognexa.cli.assistant.commands.create_knowledge_base")
    def test_add_document_propagates_exception(
        self,
        mock_create_knowledge_base,
    ):
        knowledge_base = MagicMock()
        knowledge_base.add_document.side_effect = ValueError("Invalid document")

        mock_create_knowledge_base.return_value = knowledge_base

        with pytest.raises(ValueError, match="Invalid document"):
            add_document("invalid.pdf")

        knowledge_base.add_document.assert_called_once_with("invalid.pdf")


class TestChat:

    @patch("cognexa.cli.assistant.commands.create_assistant")
    @patch("builtins.input", side_effect=["exit"])
    def test_chat_exits_on_exit_command(
        self,
        mock_input,
        mock_create_assistant,
        capsys,
    ):
        chat()

        mock_create_assistant.assert_called_once()
        mock_input.assert_called_once_with("You: ")

        captured = capsys.readouterr()

        assert "Cognexa Assistant" in captured.out
        assert "RAG mode: enabled" in captured.out

    @patch("cognexa.cli.assistant.commands.create_assistant")
    @patch("builtins.input", side_effect=["", "   ", "exit"])
    def test_chat_skips_empty_messages(
        self,
        mock_input,
        mock_create_assistant,
    ):
        assistant = mock_create_assistant.return_value

        chat()

        assistant.chat.assert_not_called()
        assert mock_input.call_count == 3

    @patch("cognexa.cli.assistant.commands.create_assistant")
    @patch("builtins.input", side_effect=["Hello", "exit"])
    def test_chat_processes_user_message(
        self,
        mock_input,
        mock_create_assistant,
        capsys,
    ):
        assistant = mock_create_assistant.return_value
        assistant.chat.return_value = "Hello! How can I help?"

        chat()

        assistant.chat.assert_called_once_with("Hello")

        captured = capsys.readouterr()

        assert "Cognexa: Hello! How can I help?" in captured.out

    @patch("cognexa.cli.assistant.commands.create_assistant")
    @patch(
        "builtins.input",
        side_effect=["Hello", "How are you?", "exit"],
    )
    def test_chat_processes_multiple_messages(
        self,
        mock_input,
        mock_create_assistant,
    ):
        assistant = mock_create_assistant.return_value
        assistant.chat.side_effect = [
            "Hello!",
            "I am fine!",
        ]

        chat()

        assert assistant.chat.call_args_list == [
            call("Hello"),
            call("How are you?"),
        ]



