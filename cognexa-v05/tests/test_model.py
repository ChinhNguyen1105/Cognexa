from cognexa.tokenizer import TextTokenizer
from cognexa.model import TransformerModel


def test_transformer_model():
    tokenizer = TextTokenizer()
    model = TransformerModel()

    inputs = tokenizer.encode(
        "Artificial intelligence is important."
    )

    representations = model.encode(inputs)

    assert representations.ndim == 3
    assert representations.shape[0] == 1
    assert representations.shape[1] == 7
    assert representations.shape[2] == 768