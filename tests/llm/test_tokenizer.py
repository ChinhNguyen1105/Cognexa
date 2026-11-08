from unittest.mock import MagicMock, patch

import torch

from cognexa.llm.tokenizer import TextTokenizer


class TestTextTokenizer:

    @patch("cognexa.llm.tokenizer.AutoTokenizer.from_pretrained")
    def test_init_loads_pretrained_tokenizer(
        self,
        mock_from_pretrained,
    ):
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = TextTokenizer("Qwen/test-model")

        mock_from_pretrained.assert_called_once_with("Qwen/test-model")
        assert tokenizer.tokenizer is mock_tokenizer

    @patch("cognexa.llm.tokenizer.AutoTokenizer.from_pretrained")
    def test_encode_uses_chat_template(
        self,
        mock_from_pretrained,
    ):
        mock_tokenizer = MagicMock()

        expected_result = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_tokenizer.apply_chat_template.return_value = expected_result
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = TextTokenizer("Qwen/test-model")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        result = tokenizer.encode(messages)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        assert result is expected_result
        assert result["input_ids"].tolist() == [[1, 2, 3]]

    @patch("cognexa.llm.tokenizer.AutoTokenizer.from_pretrained")
    def test_decode_skips_special_tokens(
        self,
        mock_from_pretrained,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Hello world"

        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = TextTokenizer("Qwen/test-model")

        token_ids = torch.tensor([1, 2, 3])

        result = tokenizer.decode(token_ids)

        mock_tokenizer.decode.assert_called_once_with(
            token_ids,
            skip_special_tokens=True,
        )

        assert result == "Hello world"
