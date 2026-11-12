from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from cognexa.llm.generator import TextGenerator


class TestTextGenerator:

    def create_generator(
        self,
        model=None,
        max_new_tokens=3,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        attention_mask=True,
    ):
        if model is None:
            model = MagicMock()

        tokenizer = MagicMock()

        input_ids = torch.tensor([[1, 2]])

        tokenizer.encode.return_value = {
            "input_ids": input_ids,
        }

        if attention_mask:
            tokenizer.encode.return_value["attention_mask"] = (
                torch.tensor([[1, 1]])
            )

        tokenizer.tokenizer.eos_token_id = 99
        tokenizer.decode.return_value = "Generated answer"

        return TextGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    def create_outputs(
        self,
        token_id=3,
        past_key_values=None,
        vocab_size=5,
    ):
        logits = torch.zeros((1, 2, vocab_size))
        logits[:, -1, token_id] = 10.0

        return SimpleNamespace(
            logits=logits,
            past_key_values=past_key_values,
        )

    def test_greedy_decoding_with_eos(self):
        model = MagicMock()
        cache = MagicMock()

        model.return_value = self.create_outputs(
            token_id=3,
            past_key_values=cache,
        )

        generator = self.create_generator(
            model=model,
            do_sample=False,
        )

        generator.tokenizer.tokenizer.eos_token_id = 3

        result = generator.generate(
            [{"role": "user", "content": "Hello"}]
        )

        model.assert_called_once()

        call_kwargs = model.call_args.kwargs

        assert call_kwargs["use_cache"] is True
        assert call_kwargs["input_ids"].tolist() == [[1, 2]]
        assert result == "Generated answer"

        generator.tokenizer.decode.assert_called_once()

        decoded_ids = generator.tokenizer.decode.call_args.args[0]

        assert decoded_ids.tolist() == [3]

    def test_generate_with_tuple_model_output(self):
        model = MagicMock()

        cache = MagicMock()

        logits = torch.zeros((1, 2, 5))
        logits[:, -1, 3] = 10.0

        model.return_value = (logits, cache)

        generator = self.create_generator(
            model=model,
            do_sample=False,
        )

        generator.tokenizer.tokenizer.eos_token_id = 3

        result = generator.generate(
            [{"role": "user", "content": "Hello"}]
        )

        assert result == "Generated answer"
        model.assert_called_once()

    def test_generate_with_repetition_penalty(self):
        model = MagicMock()

        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            do_sample=False,
            repetition_penalty=1.5,
            max_new_tokens=1,
        )

        result = generator.generate(
            [{"role": "user", "content": "Repeat"}]
        )

        assert result == "Generated answer"
        model.assert_called_once()

    @patch("torch.multinomial")
    def test_sampling_with_temperature_top_k_and_top_p(
        self,
        mock_multinomial,
    ):
        mock_multinomial.return_value = torch.tensor([[3]])

        model = MagicMock()
        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            temperature=0.5,
            top_k=3,
            top_p=0.8,
            do_sample=True,
        )

        result = generator.generate(
            [{"role": "user", "content": "Sample"}]
        )

        assert result == "Generated answer"
        mock_multinomial.assert_called_once()

    @patch("torch.multinomial")
    def test_sampling_without_temperature_scaling(
        self,
        mock_multinomial,
    ):
        mock_multinomial.return_value = torch.tensor([[3]])

        model = MagicMock()
        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            temperature=0,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )

        result = generator.generate(
            [{"role": "user", "content": "Sample"}]
        )

        assert result == "Generated answer"
        mock_multinomial.assert_called_once()

    @patch("torch.multinomial")
    def test_sampling_without_top_k(
        self,
        mock_multinomial,
    ):
        mock_multinomial.return_value = torch.tensor([[3]])

        model = MagicMock()
        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            top_k=0,
            top_p=1.0,
        )

        result = generator.generate(
            [{"role": "user", "content": "Sample"}]
        )

        assert result == "Generated answer"
        mock_multinomial.assert_called_once()

    @patch("torch.multinomial")
    def test_top_k_greater_than_vocab_size(
        self,
        mock_multinomial,
    ):
        mock_multinomial.return_value = torch.tensor([[3]])

        model = MagicMock()
        model.return_value = self.create_outputs(
            token_id=3,
            vocab_size=5,
        )

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            top_k=100,
            top_p=1.0,
        )

        result = generator.generate(
            [{"role": "user", "content": "Sample"}]
        )

        assert result == "Generated answer"
        mock_multinomial.assert_called_once()

    @patch("torch.multinomial")
    def test_sampling_without_top_p(
        self,
        mock_multinomial,
    ):
        mock_multinomial.return_value = torch.tensor([[3]])

        model = MagicMock()
        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            top_k=3,
            top_p=1.0,
        )

        result = generator.generate(
            [{"role": "user", "content": "Sample"}]
        )

        assert result == "Generated answer"
        mock_multinomial.assert_called_once()

    def test_generate_stops_when_eos_is_generated(self):
        model = MagicMock()

        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            max_new_tokens=5,
            do_sample=False,
        )

        generator.tokenizer.tokenizer.eos_token_id = 3

        generator.generate(
            [{"role": "user", "content": "Stop"}]
        )

        model.assert_called_once()

    def test_generate_without_attention_mask(self):
        model = MagicMock()

        model.return_value = self.create_outputs(token_id=3)

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            do_sample=False,
            attention_mask=False,
        )

        generator.generate(
            [{"role": "user", "content": "No mask"}]
        )

        call_kwargs = model.call_args.kwargs

        assert call_kwargs["attention_mask"] is None

    def test_generate_with_zero_max_new_tokens(self):
        model = MagicMock()

        generator = self.create_generator(
            model=model,
            max_new_tokens=0,
        )

        result = generator.generate(
            [{"role": "user", "content": "Hello"}]
        )

        model.assert_not_called()
        generator.tokenizer.decode.assert_called_once()

        assert result == "Generated answer"

    def test_generate_uses_no_grad(self):
        model = MagicMock()

        def model_forward(**kwargs):
            assert torch.is_grad_enabled() is False
            return self.create_outputs(token_id=3)

        model.side_effect = model_forward

        generator = self.create_generator(
            model=model,
            max_new_tokens=1,
            do_sample=False,
        )

        generator.generate(
            [{"role": "user", "content": "No grad"}]
        )

        model.assert_called_once()

    def test_generate_passes_cache_between_iterations(self):
        model = MagicMock()

        cache = MagicMock()

        model.side_effect = [
            self.create_outputs(
                token_id=3,
                past_key_values=cache,
            ),
            self.create_outputs(
                token_id=4,
                past_key_values=cache,
            ),
        ]

        generator = self.create_generator(
            model=model,
            max_new_tokens=2,
            do_sample=False,
        )

        generator.tokenizer.tokenizer.eos_token_id = 99

        generator.generate(
            [{"role": "user", "content": "Cache"}]
        )

        assert model.call_count == 2

        first_call = model.call_args_list[0].kwargs
        second_call = model.call_args_list[1].kwargs

        assert first_call["past_key_values"] is None
        assert second_call["past_key_values"] is cache

    def test_generate_updates_attention_mask(self):
        model = MagicMock()

        model.side_effect = [
            self.create_outputs(token_id=3),
            self.create_outputs(token_id=4),
        ]

        generator = self.create_generator(
            model=model,
            max_new_tokens=2,
            do_sample=False,
        )

        generator.tokenizer.tokenizer.eos_token_id = 99

        generator.generate(
            [{"role": "user", "content": "Attention"}]
        )

        assert model.call_count == 2

        first_mask = model.call_args_list[0].kwargs["attention_mask"]
        second_mask = model.call_args_list[1].kwargs["attention_mask"]

        assert first_mask.tolist() == [[1, 1]]
        assert second_mask.tolist() == [[1, 1, 1]]




