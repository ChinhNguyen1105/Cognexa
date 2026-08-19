from cognexa.main import DocumentPipeline


def test_document_pipeline():
    pipeline = DocumentPipeline()

    prediction = pipeline.predict(
        "Artificial intelligence is important."
    )

    assert isinstance(prediction, int)
    assert prediction in (0, 1)