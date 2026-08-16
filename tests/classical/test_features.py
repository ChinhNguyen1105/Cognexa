from cognexa.ml.classical.features import TextFeatureExtractor


def test_text_feature_extraction():
    texts = [
        "machine learning",
        "deep learning",
        "python programming",
    ]

    extractor = TextFeatureExtractor()
    features = extractor.fit_transform(texts)

    assert features.shape[0] == 3
    assert features.shape[1] > 0