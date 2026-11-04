from torch import nn
from transformers import AutoModelForCausalLM


class LLM(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        past_key_values=None, 
        use_cache: bool = False
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Trả về cả logits và past_key_values nếu use_cache=True
        if use_cache:
            return outputs.logits, outputs.past_key_values

        return outputs.logits