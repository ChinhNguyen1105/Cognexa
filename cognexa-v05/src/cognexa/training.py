from cognexa.classifier import DocumentClassifier
from cognexa.dataset import DocumentDataset
from cognexa.model import TransformerModel
from cognexa.tokenizer import TextTokenizer
import torch
from torch import nn, optim


class DocumentTrainer:
    """Train the document classification head."""

    def __init__(self, learning_rate: float = 1e-3):
        self.tokenizer = TextTokenizer()
        self.model = TransformerModel()
        self.classifier = DocumentClassifier()

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate,
        )

    def train_sample(self, text: str, label: int) -> float:
        """Train the classifier on one document."""

        inputs = self.tokenizer.encode(text)
        hidden_states = self.model.encode(inputs)

        logits = self.classifier(
            hidden_states,
            inputs["attention_mask"],
        )

        target = torch.tensor([label]) 

        loss = self.loss_function(logits, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset_path: str, epochs: int = 5) -> list[float]:
        """Train the classifier on the complete dataset."""

        dataset = DocumentDataset(dataset_path)
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0

            for sample in dataset:
                loss = self.train_sample(
                    sample["text"],
                    sample["label"].item(),
                )

                total_loss += loss

            average_loss = total_loss / len(dataset)
            losses.append(average_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.4f}"
            )

        return losses