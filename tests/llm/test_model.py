from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from cognexa.llm.model import LLM


class TestLLM:

    @patch("cognexa.llm.model.AutoModelForCausalLM.from_pretrained")
    def test_init_loads_pretrained_model(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = LLM("Qwen/test-model")

        mock_from_pretrained.assert_called_once_with("Qwen/test-model")
        assert model.model is mock_model
        assert isinstance(model, nn.Module)

    @patch("cognexa.llm.model.AutoModelForCausalLM.from_pretrained")
    def test_forward_with_cache_returns_logits_and_past_key_values(
        self,
        mock_from_pretrained,
    ):
        logits = torch.randn(1, 2, 10)
        cache = MagicMock()

        underlying_model = MagicMock()
        underlying_model.return_value = SimpleNamespace(
            logits=logits,
            past_key_values=cache,
        )

        mock_from_pretrained.return_value = underlying_model

        model = LLM("Qwen/test-model")

        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])

        result_logits, result_cache = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )

        assert result_logits is logits
        assert result_cache is cache

        underlying_model.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )

    @patch("cognexa.llm.model.AutoModelForCausalLM.from_pretrained")
    def test_forward_without_cache_returns_only_logits(
        self,
        mock_from_pretrained,
    ):
        logits = torch.randn(1, 2, 10)

        underlying_model = MagicMock()
        underlying_model.return_value = SimpleNamespace(
            logits=logits,
            past_key_values=None,
        )

        mock_from_pretrained.return_value = underlying_model

        model = LLM("Qwen/test-model")

        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])

        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        assert result is logits

        underlying_model.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )

    @patch("cognexa.llm.model.AutoModelForCausalLM.from_pretrained")
    def test_forward_with_default_arguments(
        self,
        mock_from_pretrained,
    ):
        logits = torch.randn(1, 2, 10)

        underlying_model = MagicMock()
        underlying_model.return_value = SimpleNamespace(
            logits=logits,
            past_key_values=None,
        )

        mock_from_pretrained.return_value = underlying_model

        model = LLM("Qwen/test-model")

        input_ids = torch.tensor([[1, 2]])

        result = model(input_ids=input_ids)

        assert result is logits

        underlying_model.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
        )


