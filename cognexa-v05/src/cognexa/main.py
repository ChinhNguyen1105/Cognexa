import sys

import torch

from cognexa.classifier import DocumentClassifier
from cognexa.model import TransformerModel
from cognexa.persistence import load_model
from cognexa.tokenizer import TextTokenizer


LABELS = {
    0: "AI",
    1: "Programming",
}


class DocumentPipeline:
    """End-to-end Transformer document classification pipeline."""

    def __init__(
        self,
        model_path: str | None = None,
        num_classes: int = 2,
    ):
        self.tokenizer = TextTokenizer()
        self.model = TransformerModel()
        self.classifier = DocumentClassifier(
            num_classes=num_classes,
        )

        if model_path is not None:
            load_model(
                self.classifier,
                model_path,
            )

        self.classifier.eval()

    def predict(self, text: str) -> int:
        """Predict the class of a document."""

        inputs = self.tokenizer.encode(text)

        hidden_states = self.model.encode(inputs)

        with torch.no_grad():
            logits = self.classifier(
                hidden_states,
                inputs["attention_mask"],
            )

        return torch.argmax(
            logits,
            dim=-1,
        ).item()


def main() -> None:
    """Run the Cognexa command-line interface."""

    if len(sys.argv) < 3:
        print(
            "Usage: python -m cognexa.main "
            'predict "text"'
        )
        return

    command = sys.argv[1]
    text = sys.argv[2]

    if command != "predict":
        print(f"Unknown command: {command}")
        return

    pipeline = DocumentPipeline(
        "models/document_classifier.pth"
    )

    prediction = pipeline.predict(text)

    print(f"Prediction: {LABELS[prediction]}")


if __name__ == "__main__":
    main()