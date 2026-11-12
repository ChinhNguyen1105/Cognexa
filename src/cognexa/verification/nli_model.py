import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from cognexa.verification.schemas import NLIScores


class NLIModel:
    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 512,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")

        self.model_name = model_name
        self.max_length = max_length

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()
        self.label_indices = self._resolve_label_indices()

    def _resolve_label_indices(self) -> dict[str, int]:
        label_map = {
            str(label).lower(): int(index)
            for index, label in self.model.config.id2label.items()
        }

        aliases = {
            "entailment": ("entailment",),
            "contradiction": ("contradiction",),
            "neutral": ("neutral",),
        }

        resolved = {}

        for target, names in aliases.items():
            for label, index in label_map.items():
                if any(name in label for name in names):
                    resolved[target] = index
                    break

            if target not in resolved:
                raise ValueError(
                    f"Cannot resolve NLI label: {target}. "
                    f"Available labels: {label_map}"
                )

        if len(set(resolved.values())) != 3:
            raise ValueError("NLI labels must map to distinct indices.")

        return resolved

    @torch.inference_mode()
    def predict_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
    ) -> list[NLIScores]:
        if len(premises) != len(hypotheses):
            raise ValueError(
                "Premises and hypotheses must have equal lengths."
            )

        if not premises:
            return []

        inputs = premises + hypotheses

        if any(not isinstance(text, str) for text in inputs):
            raise TypeError("NLI inputs must be strings.")

        if any(not text.strip() for text in inputs):
            raise ValueError("NLI inputs must not be empty.")

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits.float(), dim=-1)

        results = []

        for row in probabilities:
            results.append(
                NLIScores(
                    entailment=float(
                        row[self.label_indices["entailment"]].item()
                    ),
                    contradiction=float(
                        row[self.label_indices["contradiction"]].item()
                    ),
                    neutral=float(
                        row[self.label_indices["neutral"]].item()
                    ),
                )
            )

        return results

    def predict(
        self,
        premise: str,
        hypothesis: str,
    ) -> NLIScores:
        return self.predict_batch(
            [premise],
            [hypothesis],
        )[0]

