from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from cognexa.verification import SentenceStatus
from cognexa.verification.preprocessing import TextPreprocessor


@dataclass

class SelectionResult:
    sentences: list[dict[str, str]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)
    uncovered_aspects: list[str] = field(default_factory=list)
    top_relevance: float | None = None


class AnswerSelector:
    def __init__(
        self,
        answer_verifier=None,
        min_relevance_score: float | None = None,
        max_sentences: int = 5,
        redundancy_threshold: float = 0.7,
    ):
        if max_sentences <= 0:
            raise ValueError("max_sentences must be greater than zero")
        if not 0.0 <= redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be between 0 and 1")

        self.answer_verifier = answer_verifier
        self.min_relevance_score = min_relevance_score
        self.max_sentences = max_sentences
        self.redundancy_threshold = redundancy_threshold

    @staticmethod
    def _normalize_tokens(text: str) -> set[str]:
        if not isinstance(text, str):
            return set()
        normalized = text.replace("-", " ")
        return set(re.findall(r"[A-Za-z0-9]+", normalized.casefold()))

    @staticmethod
    def _normalized_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return " ".join(re.findall(r"[A-Za-z0-9]+", text.casefold()))

    @staticmethod
    def _jaccard_similarity(left: str, right: str) -> float:
        left_tokens = AnswerSelector._normalize_tokens(left)
        right_tokens = AnswerSelector._normalize_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = left_tokens & right_tokens
        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            return len(intersection) / min(len(left_tokens), len(right_tokens)) if min(len(left_tokens), len(right_tokens)) else 0.0
        union = left_tokens | right_tokens
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _is_redundant(left: str, right: str, threshold: float) -> bool:
        left_norm = AnswerSelector._normalized_text(left)
        right_norm = AnswerSelector._normalized_text(right)
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        if left_norm in right_norm or right_norm in left_norm:
            return True
        return AnswerSelector._jaccard_similarity(left, right) >= threshold

    @staticmethod
    def _new_info_ratio(sentence: str, reference: str) -> float:
        sentence_tokens = AnswerSelector._normalize_tokens(sentence)
        reference_tokens = AnswerSelector._normalize_tokens(reference)
        if not sentence_tokens:
            return 0.0
        if not reference_tokens:
            return 1.0
        new_tokens = sentence_tokens - reference_tokens
        return len(new_tokens) / len(sentence_tokens)

    @staticmethod
    def _should_keep_sentence(sentence: str) -> bool:
        tokens = AnswerSelector._normalize_tokens(sentence)
        if not tokens:
            return False
        if len(tokens) <= 2:
            return False
        content_tokens = {token for token in tokens if len(token) > 2}
        return bool(content_tokens)

    @staticmethod
    def _looks_like_continuation(sentence: str) -> bool:
        text = " ".join(re.findall(r"[A-Za-z0-9]+", (sentence or "").casefold()))
        if not text:
            return False
        starters = (
            "common approaches include",
            "for example",
            "this",
            "these",
            "those",
            "as ",
            "in addition",
            "furthermore",
            "moreover",
            "another",
            "also",
            "it",
        )
        return any(text.startswith(starter) for starter in starters)

    @staticmethod
    def _clean_context(context: str) -> list[str]:
        if not isinstance(context, str) or not context.strip():
            return []

        return TextPreprocessor.split_evidence(context)

    def _compose_adjacent_selection(
        self,
        selected: list[dict[str, Any]],
        supported: list[dict[str, Any]],
        evidence: list[str],
        seen_texts: set[str],
    ) -> None:
        if not selected or not evidence:
            return

        by_normalized = {
            self._normalized_text(item["sentence"]): item
            for item in supported
            if isinstance(item.get("sentence"), str)
        }

        for item in list(selected):
            sentence = item["sentence"]
            normalized = self._normalized_text(sentence)
            if not normalized:
                continue

            match_index = next(
                (
                    idx
                    for idx, candidate in enumerate(evidence)
                    if self._normalized_text(candidate) == normalized
                ),
                None,
            )
            if match_index is None or match_index + 1 >= len(evidence):
                continue

            next_sentence = evidence[match_index + 1]
            next_norm = self._normalized_text(next_sentence)
            if not next_norm or next_norm in seen_texts:
                continue

            if not self._looks_like_continuation(next_sentence):
                continue

            next_item = by_normalized.get(next_norm)
            if next_item is None:
                continue

            if any(
                self._is_redundant(next_item["sentence"], existing["sentence"], self.redundancy_threshold)
                for existing in selected
            ):
                continue

            selected.append(next_item)
            seen_texts.add(next_item["sentence"])

    def select(self, candidates, context: str) -> list[str]:
        """Compatibility wrapper returning selected sentence texts."""
        result = self.select_detailed(
            candidates=candidates,
            context=context,
        )
        return result.texts

    def select_detailed(self, candidates, context: str) -> SelectionResult:
        if not candidates:
            return SelectionResult()
        if not isinstance(context, str) or not context.strip():
            return SelectionResult()

        evidence = self._clean_context(context)
        if not evidence:
            return SelectionResult()

        normalized = []
        for item in candidates:
            if isinstance(item, dict):
                sentence = item.get("sentence")
                score = item.get("score", 0.0)
                covered = list(item.get("covered_aspects", []) or [])
                aspect_scores = dict(item.get("aspect_scores", {}) or {})
            else:
                sentence = item
                score = 0.0
                covered = []
                aspect_scores = {}

            if not isinstance(sentence, str) or not sentence.strip():
                continue

            if not self._should_keep_sentence(sentence):
                continue

            normalized.append({
                "sentence": sentence,
                "score": float(score),
                "covered_aspects": covered,
                "aspect_scores": aspect_scores,
            })

        if not normalized:
            return SelectionResult()

        normalized.sort(key=lambda item: item["score"], reverse=True)

        all_aspects: set[str] = set()
        supported: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for item in normalized:
            for aspect in item["covered_aspects"]:
                if isinstance(aspect, str):
                    all_aspects.add(aspect)
            for aspect in item["aspect_scores"]:
                if isinstance(aspect, str) and not aspect.startswith("q-"):
                    all_aspects.add(aspect)
            if self.answer_verifier is None:
                supported.append(item)
                continue

            verification = self.answer_verifier.verify(item["sentence"], evidence)
            statuses = getattr(verification, "sentences", []) or []
            if not statuses or all(getattr(status, "status", SentenceStatus.SUPPORTED) == SentenceStatus.SUPPORTED for status in statuses):
                supported.append(item)
            else:
                failed = next(
                    (
                        getattr(status, "status", SentenceStatus.SUPPORTED)
                        for status in statuses
                        if getattr(status, "status", SentenceStatus.SUPPORTED) != SentenceStatus.SUPPORTED
                    ),
                    SentenceStatus.UNSUPPORTED,
                )
                rejected.append({
                    "sentence": item["sentence"],
                    "status": failed.value if hasattr(failed, "value") else str(failed),
                    "relevance": item["score"],
                })

        if not supported:
            return SelectionResult(
                rejected=rejected,
                aspects=sorted(all_aspects),
                top_relevance=max((item["score"] for item in normalized), default=None),
            )

        selected: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        selected_aspects: set[str] = set()
        aspect_best: dict[str, dict[str, Any]] = {}
        for item in supported:
            for aspect in item["covered_aspects"]:
                if not isinstance(aspect, str):
                    continue
                current = aspect_best.get(aspect)
                if current is None or item["score"] > current["score"]:
                    aspect_best[aspect] = item

        aspect_order = sorted(aspect_best.items(), key=lambda kv: kv[1]["score"], reverse=True)
        for aspect, item in aspect_order:
            if len(selected) >= self.max_sentences:
                break
            text = item["sentence"]
            if text in seen_texts:
                continue
            if any(self._is_redundant(text, existing["sentence"], self.redundancy_threshold) for existing in selected):
                continue
            selected.append(item)
            seen_texts.add(text)
            selected_aspects.add(aspect)

        for item in supported:
            text = item["sentence"]
            if text in seen_texts:
                continue
            if len(selected) >= self.max_sentences:
                break
            if self.min_relevance_score is not None and item["score"] < self.min_relevance_score:
                continue
            if any(self._is_redundant(text, existing["sentence"], self.redundancy_threshold) for existing in selected):
                continue
            selected.append(item)
            seen_texts.add(text)

        self._compose_adjacent_selection(selected, supported, evidence, seen_texts)
        if len(selected) > self.max_sentences:
            selected = selected[: self.max_sentences]

        full_texts = [item["sentence"] for item in selected]
        uncovered = sorted(all_aspects - selected_aspects)
        top_relevance = max((item["score"] for item in normalized if item["sentence"] in full_texts), default=None)

        if top_relevance is None and supported:
            top_relevance = max(item["score"] for item in supported)

        return SelectionResult(
            sentences=[{"sentence": text} for text in full_texts],
            texts=full_texts,
            rejected=rejected,
            aspects=sorted(all_aspects),
            uncovered_aspects=uncovered,
            top_relevance=top_relevance,
        )



