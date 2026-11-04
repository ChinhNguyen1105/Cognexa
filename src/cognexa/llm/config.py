from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    max_new_tokens: int = 128

    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

    repetition_penalty: float = 1.1