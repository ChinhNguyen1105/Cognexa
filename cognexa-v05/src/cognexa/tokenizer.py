from transformers import AutoTokenizer


class TextTokenizer:
    """Tokenize text using a pretrained Hugging Face tokenizer."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str):
        """Convert text into model-ready tokens."""
        return self.tokenizer(
            text,
            return_tensors="pt",
        )