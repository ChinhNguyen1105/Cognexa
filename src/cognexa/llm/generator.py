
import torch

from .model import LLM
from .tokenizer import TextTokenizer


class TextGenerator:
    """
    Generate text from an instruction-tuned LLM
    using controlled sampling and KV Caching.
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        inputs = self.tokenizer.encode(messages)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        prompt_length = input_ids.size(1)
        eos_token_id = self.tokenizer.tokenizer.eos_token_id

        # Khởi tạo KV cache
        past_key_values = None
        current_input_ids = input_ids
        
        # Mảng lưu trữ tất cả token (prompt + generated)
        all_token_ids = input_ids

        for _ in range(self.max_new_tokens):
            # Truyền past_key_values vào mô hình (chỉ xử lý token mới nhất nếu past_key_values != None)
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Lấy logits và KV cache mới (tùy thuộc vào output structure của model)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                logits, past_key_values = outputs[0], outputs[1]

            next_token_logits = logits[:, -1, :].clone()

            # Vectorized Repetition Penalty
            if self.repetition_penalty > 1.0:
                score = torch.gather(next_token_logits, 1, all_token_ids)
                # Tăng penalty cho logits âm, giảm cho logits dương
                score = torch.where(
                    score < 0, 
                    score * self.repetition_penalty, 
                    score / self.repetition_penalty
                )
                next_token_logits.scatter_(1, all_token_ids, score)

            # Temperature
            if self.temperature > 0:
                next_token_logits = next_token_logits / self.temperature

            # Top-k filtering
            if self.top_k > 0:
                top_k = min(self.top_k, next_token_logits.size(-1))
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Top-p (Nucleus) filtering
            if 0.0 < self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                cumulative_probabilities = torch.cumsum(
                    sorted_probabilities, dim=-1
                )

                sorted_indices_to_remove = (
                    cumulative_probabilities > self.top_p
                )
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = torch.zeros_like(
                    next_token_logits, dtype=torch.bool
                )
                indices_to_remove.scatter_(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf")
                )

            # Sample next token
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Cập nhật mảng lưu trữ token
            all_token_ids = torch.cat([all_token_ids, next_token], dim=1)

            # Chuẩn bị cho bước lặp tiếp theo (chỉ truyền duy nhất 1 token mới)
            current_input_ids = next_token

            # Cập nhật attention_mask nếu dùng
            if attention_mask is not None:
                next_attention = torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    [attention_mask, next_attention], dim=1
                )

            # Dừng lại nếu gặp token EOS
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        # Tách phần token được sinh ra
        generated_token_ids = all_token_ids[:, prompt_length:]

        return self.tokenizer.decode(generated_token_ids[0]).strip()