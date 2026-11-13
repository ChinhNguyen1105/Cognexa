class PromptBuilder:
    def __init__(self, system_prompt: str = "You are a precise answering assistant. Formulate your answer STRICTLY based on the provided context. If the context contains a list of reasons, methods, or factors, you must comprehensively synthesize all of them in your answer. Do not include external knowledge. If the context does not contain the answer, state that you do not know."):
        self.system_prompt = system_prompt or "You are a precise answering assistant. Formulate your answer STRICTLY based on the provided context. If the context contains a list of reasons, methods, or factors, you must comprehensively synthesize all of them in your answer. Do not include external knowledge. If the context does not contain the answer, state that you do not know."

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
