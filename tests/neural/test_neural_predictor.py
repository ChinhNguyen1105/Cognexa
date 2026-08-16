
import torch

from cognexa.ml.neural.inference.predictor import NeuralPredictor
from cognexa.ml.neural.models.document_classifier import DocumentClassifier


def test_neural_predictor_returns_class_predictions():
    model = DocumentClassifier(
        input_size=2,
        hidden_size=4,
        num_classes=2,
    )

    predictor = NeuralPredictor(model)

    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    predictions = predictor.predict(features)

    assert predictions.shape == (3,)
    assert predictions.dtype == torch.int64
    assert all(prediction in (0, 1) for prediction in predictions.tolist())
