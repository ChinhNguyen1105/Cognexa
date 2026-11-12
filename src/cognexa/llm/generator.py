import torch

from .model import LLM
from .tokenizer import TextTokenizer


class TextGenerator:
    """
    Generate text from an instruction-tuned LLM
    using controlled decoding and KV Caching.
    """

    def __init__(
        self,
        model: LLM,
        tokenizer: TextTokenizer,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

        inputs = self.tokenizer.encode(messages)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        prompt_length = input_ids.size(1)
        eos_token_id = self.tokenizer.tokenizer.eos_token_id

        past_key_values = None
        current_input_ids = input_ids
        all_token_ids = input_ids

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if hasattr(outputs, "logits"):
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                logits, past_key_values = outputs[0], outputs[1]

            next_token_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if self.repetition_penalty > 1.0:
                score = torch.gather(
                    next_token_logits,
                    1,
                    all_token_ids,
                )

                score = torch.where(
                    score < 0,
                    score * self.repetition_penalty,
                    score / self.repetition_penalty,
                )

                next_token_logits.scatter_(
                    1,
                    all_token_ids,
                    score,
                )

            if self.do_sample:
                # Temperature
                if self.temperature > 0:
                    next_token_logits = (
                        next_token_logits / self.temperature
                    )

                # Top-k filtering
                if self.top_k > 0:
                    top_k = min(
                        self.top_k,
                        next_token_logits.size(-1),
                    )

                    top_k_values, _ = torch.topk(
                        next_token_logits,
                        top_k,
                    )

                    min_top_k = top_k_values[:, -1].unsqueeze(-1)

                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(
                            next_token_logits,
                            float("-inf"),
                        ),
                        next_token_logits,
                    )

                # Top-p filtering
                if 0.0 < self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits,
                        descending=True,
                    )

                    sorted_probabilities = torch.softmax(
                        sorted_logits,
                        dim=-1,
                    )

                    cumulative_probabilities = torch.cumsum(
                        sorted_probabilities,
                        dim=-1,
                    )

                    sorted_indices_to_remove = (
                        cumulative_probabilities > self.top_p
                    )

                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = torch.zeros_like(
                        next_token_logits,
                        dtype=torch.bool,
                    )

                    indices_to_remove.scatter_(
                        1,
                        sorted_indices,
                        sorted_indices_to_remove,
                    )

                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove,
                        float("-inf"),
                    )

                probabilities = torch.softmax(
                    next_token_logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            else:
                # Greedy decoding
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True,
                )

            all_token_ids = torch.cat(
                [all_token_ids, next_token],
                dim=1,
            )

            current_input_ids = next_token

            if attention_mask is not None:
                next_attention = torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                attention_mask = torch.cat(
                    [attention_mask, next_attention],
                    dim=1,
                )

            if (
                eos_token_id is not None
                and torch.all(next_token == eos_token_id)
            ):
                break

        generated_token_ids = all_token_ids[:, prompt_length:]

        return self.tokenizer.decode(
            generated_token_ids[0]
        ).strip()

