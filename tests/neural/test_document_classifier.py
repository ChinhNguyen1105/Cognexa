
import torch

from cognexa.ml.neural.models.document_classifier import DocumentClassifier


def test_document_classifier_output_shape():
    model = DocumentClassifier(
        input_size=10,
        hidden_size=8,
        num_classes=3,
    )

    x = torch.randn(4, 10)

    output = model(x)

    assert output.shape == (4, 3)
