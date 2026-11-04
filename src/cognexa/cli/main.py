import argparse

from cognexa.cli.assistant.commands import chat


def main():
    parser = argparse.ArgumentParser(
        prog="cognexa",
        description="Cognexa AI Platform",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

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

    args = parser.parse_args()

    if (
        args.command == "assistant"
        and args.assistant_command == "chat"
    ):
        chat()
        return

    parser.print_help()


if __name__ == "__main__":
    main()