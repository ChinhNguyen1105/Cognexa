class Conversation:
    def __init__(self, history=None):
        self._history = list(history or [])

    def add_user_message(self, message: str) -> None:
        self._history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self._history.append({"role": "assistant", "content": message})

    def get_history(self):
        return [entry.copy() for entry in self._history]

    def clear(self) -> None:
        self._history.clear()
