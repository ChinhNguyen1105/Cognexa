from cognexa.llm.generator import TextGenerator

from .conversation import Conversation
from .prompt import PromptBuilder


class Assistant:
    """
    Coordinate conversation, prompt building,
    and LLM text generation.
    """

    def __init__(
        self,
        generator: TextGenerator,
        prompt_builder: PromptBuilder,
        conversation: Conversation | None = None,
    ):
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.conversation = conversation or Conversation()

    def chat(self, user_message: str) -> str:
        history = self.conversation.get_history()

        prompt = self.prompt_builder.build(
            history=history,
            user_message=user_message,
        )

        response = self.generator.generate(prompt)

        self.conversation.add_user_message(user_message)
        self.conversation.add_assistant_message(response)

        return response

    def clear_conversation(self):
        self.conversation.clear()