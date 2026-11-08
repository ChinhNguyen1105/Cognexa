import re
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class AnswerVerifier:
    """
    Verify whether answer sentences are supported by retrieved context.

    Verification pipeline:
    1. Exact Match
    2. Evidence Matching
    3. NLI Verification on individual evidence sentences
    """

    DEFAULT_MODEL = (
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        min_entailment_score: float = 0.70,
        device: str = "cpu",
        uncertain_margin: float = 0.15,
    ):
        if not 0.0 <= min_entailment_score <= 1.0:
            raise ValueError(
                "min_entailment_score must be between 0 and 1."
            )

        if not 0.0 <= uncertain_margin <= 1.0:
            raise ValueError(
                "uncertain_margin must be between 0 and 1."
            )

        self.model_name = model_name
        self.min_entailment_score = min_entailment_score
        self.uncertain_margin = uncertain_margin
        self.device = torch.device(device)

        self.tokenizer = None
        self.model = None

        self.entailment_label_id = None
        self.contradiction_label_id = None

    def _load_model(self):
        if self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        labels = self.model.config.id2label

        for label_id, label_name in labels.items():
            normalized_label = label_name.lower()

            if normalized_label == "entailment":
                self.entailment_label_id = int(label_id)

            elif normalized_label == "contradiction":
                self.contradiction_label_id = int(label_id)

        if self.entailment_label_id is None:
            raise ValueError(
                "The NLI model does not contain an entailment label."
            )

        if self.contradiction_label_id is None:
            raise ValueError(
                "The NLI model does not contain a contradiction label."
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for exact evidence matching.
        """

        text = text.lower().strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = re.sub(
            r"[^\w\s]",
            "",
            text,
        )

        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into individual sentences.
        """

        parts = re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip(),
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    def _extract_evidence(
        self,
        context: str,
    ) -> list[str]:
        """
        Extract individual evidence sentences from context.

        Document headers are excluded.
        Duplicate evidence is removed.
        """

        sentences = self._split_sentences(context)

        evidence = []
        seen = set()

        for sentence in sentences:
            if sentence.startswith("[Document "):
                continue

            normalized = self._normalize_text(sentence)

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            evidence.append(sentence)

        return evidence

    def _find_exact_evidence(
        self,
        context: str,
        hypothesis: str,
    ) -> str | None:
        """
        Find an exact normalized sentence in the context.
        """

        normalized_hypothesis = self._normalize_text(
            hypothesis
        )

        if not normalized_hypothesis:
            return None

        for evidence in self._extract_evidence(context):
            normalized_evidence = self._normalize_text(
                evidence
            )

            if normalized_hypothesis == normalized_evidence:
                return evidence

        return None

    def _get_nli_scores(
        self,
        context: str,
        hypothesis: str,
    ) -> dict[str, float]:
        """
        Calculate NLI probabilities for one evidence sentence.
        """

        self._load_model()

        inputs = self.tokenizer(
            context,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        scores = probabilities[0]

        return {
            self.model.config.id2label[index].lower(): score.item()
            for index, score in enumerate(scores)
        }

    def _get_entailment_score(
        self,
        context: str,
        hypothesis: str,
        ) -> float:
        """
        Calculate entailment probability for a context-hypothesis pair.
        """

        self._load_model()

        inputs = self.tokenizer(
            context,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        return probabilities[
            0,
            self.entailment_label_id,
        ].item()

    def verify_sentence(
        self,
        context: str,
        hypothesis: str,
    ) -> dict[str, Any]:
        """
        Verify one answer sentence against retrieved context.

        Returns:
            {
                "status": "supported" | "unsupported" | "uncertain",
                "method": "exact_match" | "nli" | "none",
                "score": float | None,
                "evidence": str | None
            }
        """

        if not context.strip() or not hypothesis.strip():
            return {
                "status": self.UNSUPPORTED,
                "method": "none",
                "score": None,
                "evidence": None,
            }

        # Step 1: Exact Match
        exact_evidence = self._find_exact_evidence(
            context=context,
            hypothesis=hypothesis,
        )

        if exact_evidence is not None:
            return {
                "status": self.SUPPORTED,
                "method": "exact_match",
                "score": 1.0,
                "evidence": exact_evidence,
            }

        # Step 2: Evidence Matching
        evidence_sentences = self._extract_evidence(context)

        if not evidence_sentences:
            return {
                "status": self.UNSUPPORTED,
                "method": "none",
                "score": None,
                "evidence": None,
            }

        # Step 3: NLI on individual evidence sentences
        best_score = 0.0
        best_evidence = None
        best_contradiction_score = 0.0

        for evidence in evidence_sentences:
            scores = self._get_nli_scores(
                context=evidence,
                hypothesis=hypothesis,
            )

            entailment_score = scores["entailment"]

            print(
            f"[DEBUG NLI] Entailment: {entailment_score:.4f} | "
            f"Threshold: {self.min_entailment_score:.4f} | "
            f"Uncertain threshold: "
            f"{self.min_entailment_score - self.uncertain_margin:.4f}"
)
            contradiction_score = scores["contradiction"]

            if entailment_score > best_score:
                best_score = entailment_score
                best_evidence = evidence
                best_contradiction_score = contradiction_score

        # Step 4: Verification Decision
        if best_score >= self.min_entailment_score:
            status = self.SUPPORTED

        elif (
            best_score
            >= self.min_entailment_score - self.uncertain_margin
        ):
            status = self.UNCERTAIN

        else:
            status = self.UNSUPPORTED

        return {
            "status": status,
            "method": "nli",
            "score": best_score,
            "contradiction_score": best_contradiction_score,
            "evidence": best_evidence,
        }

    def get_supported_sentences(
        self,
        context: str,
        answer: str,
    ) -> list[str]:
        """
        Return only sentences with SUPPORTED status.
        """

        if not context.strip() or not answer.strip():
            return []

        sentences = self._split_sentences(answer)

        supported_sentences = []

        for sentence in sentences:
            result = self.verify_sentence(
                context=context,
                hypothesis=sentence,
            )

            if result["status"] == self.SUPPORTED:
                supported_sentences.append(sentence)

        return supported_sentences

    def is_supported(
        self,
        context: str,
        answer: str,
    ) -> bool:
        """
        Return True only when every answer sentence is supported.
        """

        if not context.strip() or not answer.strip():
            return False

        sentences = self._split_sentences(answer)

        if not sentences:
            return False

        return all(
            self.verify_sentence(
                context=context,
                hypothesis=sentence,
            )["status"] == self.SUPPORTED
            for sentence in sentences
        )


if __name__ == "__main__":
    verifier = AnswerVerifier(
        min_entailment_score=0.70,
        uncertain_margin=0.15,
    )

    test_cases = [
        {
            "name": "Semantic Entailment",
            "context": (
                "Regularization techniques help reduce overfitting "
                "by discouraging overly complex models."
            ),
            "hypothesis": (
                "Applying regularization can prevent a model from "
                "becoming excessively complex and overfitting."
            ),
        },
        {
            "name": "Neutral",
            "context": (
                "Regularization techniques help reduce overfitting "
                "by discouraging overly complex models."
            ),
            "hypothesis": (
                "Increasing the number of training epochs always "
                "prevents overfitting."
            ),
        },
        {
            "name": "Contradiction",
            "context": (
                "Regularization techniques help reduce overfitting "
                "by discouraging overly complex models."
            ),
            "hypothesis": (
                "Regularization techniques increase model complexity "
                "and make overfitting worse."
            ),
        },
    ]

    for test in test_cases:
        result = verifier.verify_sentence(
            context=test["context"],
            hypothesis=test["hypothesis"],
        )

        print(f"\n{'=' * 60}")
        print(f"TEST: {test['name']}")
        print(f"STATUS: {result['status']}")
        print(f"METHOD: {result['method']}")
        print(f"SCORE: {result['score']:.4f}")
        print(f"CONTRADICTION: {result['contradiction_score']:.4f}")
        print(f"EVIDENCE: {result['evidence']}")