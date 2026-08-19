import torch

from cognexa.classifier import DocumentClassifier


def test_document_classifier():
    classifier = DocumentClassifier()

    hidden_states = torch.randn(1, 7, 768)

    attention_mask = torch.ones(1, 7)

    logits = classifier(
        hidden_states,
        attention_mask,
    )

    assert logits.shape == (1, 2)