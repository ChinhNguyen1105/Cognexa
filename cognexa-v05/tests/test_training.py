from cognexa.training import DocumentTrainer


def test_train_dataset():
    trainer = DocumentTrainer()

    losses = trainer.train(
        "data/documents.csv",
        epochs=2,
    )

    assert len(losses) == 2
    assert all(loss >= 0 for loss in losses)