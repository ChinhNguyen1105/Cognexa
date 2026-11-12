import argparse

from cognexa.cli.assistant.commands import chat, add_document


def main():
    parser = argparse.ArgumentParser(
        prog="cognexa",
        description="Cognexa AI Platform",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # Assistant commands
    assistant_parser = subparsers.add_parser(
        "assistant",
        help="AI Assistant commands",
    )

    assistant_subparsers = assistant_parser.add_subparsers(
        dest="assistant_command"
    )

    assistant_subparsers.add_parser(
        "chat",
        help="Start an interactive AI conversation",
    )

    # Knowledge commands
    knowledge_parser = subparsers.add_parser(
        "knowledge",
        help="Knowledge base commands",
    )

    knowledge_subparsers = knowledge_parser.add_subparsers(
        dest="knowledge_command"
    )

    add_parser = knowledge_subparsers.add_parser(
        "add",
        help="Add a document to the knowledge base",
    )

    add_parser.add_argument(
        "file_path",
        type=str,
        help="Path to the document (TXT, PDF, DOCX)",
    )

    args = parser.parse_args()

    if (
        args.command == "assistant"
        and args.assistant_command == "chat"
    ):
        chat()
        return

    if (
        args.command == "knowledge"
        and args.knowledge_command == "add"
    ):
        add_document(args.file_path)
        return

    parser.print_help()


if __name__ == "__main__":
    main()


