from cognexa.ml.trainer import DocumentTrainer


def test_document_trainer():
    texts = [
        "machine learning algorithms",
        "deep neural networks",
        "python programming",
        "python functions",
    ]

    labels = [
        "ai",
        "ai",
        "programming",
        "programming",
    ]

    trainer = DocumentTrainer()
    trainer.train(texts, labels)

    predictions = trainer.predict(
        ["machine learning model"]
    )

    assert predictions[0] == "ai"