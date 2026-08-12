from src.processing.processor import process


def test_process_document():
    text = "  Hello    Cognexa\n\n World  "

    result = process(text)

    assert result == "Hello Cognexa World"

def test_process_empty_text():
    result = process("")

    assert result == ""


def test_process_whitespace_only():
    result = process("   \n\n   ")

    assert result == ""