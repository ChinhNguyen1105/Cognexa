class Conversation:
    """
    Manage the conversation history between the user and the assistant.
    """

    def __init__(self):
        self.messages: list[dict[str, str]] = []

    def add_user_message(self, content: str):
        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(self, content: str):
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    def get_history(self) -> list[dict[str, str]]:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()