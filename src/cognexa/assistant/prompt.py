class PromptBuilder:
    def __init__(self, system_prompt: str = "You are Cognexa, a helpful AI assistant."):
        self.system_prompt = system_prompt or "You are Cognexa, a helpful AI assistant."

    def build(self, history, user_message: str, context: str | None = None):
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string.")

        system_message = self.system_prompt
        if context and isinstance(context, str) and context.strip():
            system_message = (
                f"{self.system_prompt}\n\n"
                "Retrieved context:\n"
                f"{context}\n\n"
                "Use the retrieved context to answer the user and clearly state any uncertainty."
            )

        messages = [{"role": "system", "content": system_message}]

        for entry in history or []:
            messages.append({
                "role": entry.get("role", "user"),
                "content": entry.get("content", ""),
            })

        messages.append({"role": "user", "content": user_message})
        return messages
