from cognexa.evaluation import evaluate


def test_evaluate():
    accuracy = evaluate(
        "models/document_classifier.pth",
        "data/documents.csv",
    )

    assert 0.0 <= accuracy <= 1.0