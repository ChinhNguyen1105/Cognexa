from cognexa.dataset import DocumentDataset
from cognexa.main import DocumentPipeline


def evaluate(
    model_path: str,
    dataset_path: str,
) -> float:
    """Evaluate a trained document classifier."""

    pipeline = DocumentPipeline(model_path)
    dataset = DocumentDataset(dataset_path)

    correct = 0

    for sample in dataset:
        prediction = pipeline.predict(sample["text"])
        label = sample["label"].item()

        if prediction == label:
            correct += 1

    return correct / len(dataset)