class PromptBuilder:
    """
    Build structured chat messages for an instruction-tuned LLM.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build(
        self,
        history: list[dict[str, str]],
        user_message: str,
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages