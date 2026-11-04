from transformers import AutoTokenizer


class TextTokenizer:
    """
    Tokenize structured chat messages using
    the model's native chat template.
    """

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

    def encode(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )