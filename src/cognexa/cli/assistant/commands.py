from cognexa.assistant.assistant import Assistant
from cognexa.assistant.prompt import PromptBuilder
from cognexa.llm.config import LLMConfig
from cognexa.llm.generator import TextGenerator
from cognexa.llm.model import LLM
from cognexa.llm.tokenizer import TextTokenizer


def create_assistant() -> Assistant:
    config = LLMConfig()

    tokenizer = TextTokenizer(
        config.model_name
    )

    model = LLM(
        config.model_name
    )

    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )

    prompt_builder = PromptBuilder(
        system_prompt=(
            "You are Cognexa, a helpful AI assistant."
        )
    )

    return Assistant(
        generator=generator,
        prompt_builder=prompt_builder,
    )


def chat():
    assistant = create_assistant()

    print("Cognexa Assistant")
    print("Type 'exit' to quit.")
    print()

    while True:
        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        if not user_message.strip():
            continue

        response = assistant.chat(
            user_message
        )

        print(f"Cognexa: {response}")
        print()