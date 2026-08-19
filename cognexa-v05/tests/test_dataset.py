from cognexa.dataset import DocumentDataset


def test_document_dataset():
    dataset = DocumentDataset("data/documents.csv")

    assert len(dataset) == 8

    sample = dataset[0]

    assert "text" in sample
    assert "label" in sample
    assert sample["label"].item() == 0