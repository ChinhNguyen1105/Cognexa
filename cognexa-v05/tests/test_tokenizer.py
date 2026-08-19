from cognexa.tokenizer import TextTokenizer


def test_tokenize_text():
    tokenizer = TextTokenizer()

    result = tokenizer.encode(
        "Artificial intelligence is important."
    )

    assert "input_ids" in result
    assert "attention_mask" in result