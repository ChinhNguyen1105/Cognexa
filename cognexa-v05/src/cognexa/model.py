import torch
from transformers import AutoModel


class TransformerModel:
    """Load a pretrained Transformer model for text representation."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)

        # V0.5 uses the pretrained Transformer as a frozen encoder.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def encode(self, inputs):
        """Convert tokenized inputs into Transformer representations."""
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state