from __future__ import annotations

import re
from typing import Any

from cognexa.verification.preprocessing import TextPreprocessor


class AnswerExtractor:
    def __init__(
        self,
        reranker=None,
        min_candidates: int = 1,
        max_candidates: int = 10,
        score_gap_threshold: float = 0.0,
        min_score: float | None = 0.0,
    ):
        if min_candidates <= 0:
            raise ValueError("min_candidates must be greater than zero")

        if max_candidates < min_candidates:
            raise ValueError(
                "max_candidates must be greater than or equal to min_candidates"
            )

        if score_gap_threshold < 0:
            raise ValueError("score_gap_threshold must be non-negative")

        self.reranker = reranker
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.score_gap_threshold = score_gap_threshold
        self.min_score = min_score

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        return TextPreprocessor.split_evidence(text)
    @staticmethod
    def _cosine_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        left = set(re.findall(r"\b\w+\b", a.casefold()))
        right = set(re.findall(r"\b\w+\b", b.casefold()))

        if not left or not right:
            return 0.0

        return len(left & right) / max(len(left | right), 1)

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were",
            "be", "been", "being", "do", "does", "did",
            "how", "what", "why", "when", "where", "which",
            "who", "can", "could", "may", "might", "will",
            "would", "should", "to", "of", "in", "on", "at",
            "for", "from", "with", "by", "and", "or", "as",
            "it", "this", "that", "these", "those",
        }
        return {
            token
            for token in re.findall(r"\b[a-z0-9]+\b", text.casefold())
            if token not in stop_words
        }

    @classmethod
    def _query_overlap_ratio(cls, question: str, sentence: str) -> float:
        query_tokens = cls._content_tokens(question)
        sentence_tokens = cls._content_tokens(sentence)

        if not query_tokens or not sentence_tokens:
            return 0.0

        return len(query_tokens & sentence_tokens) / len(query_tokens)

    @classmethod
    def _jaccard_similarity(cls, question: str, sentence: str) -> float:
        question_tokens = cls._content_tokens(question)
        sentence_tokens = cls._content_tokens(sentence)

        if not question_tokens or not sentence_tokens:
            return 0.0

        intersection = question_tokens & sentence_tokens
        union = question_tokens | sentence_tokens

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _information_bonus(sentence: str) -> float:
        patterns = (
            r"\bbecause\b",
            r"\bdue to\b",
            r"\bwhen\b",
            r"\bmeans\b",
            r"\bwhile\b",
            r"\bresults? in\b",
            r"\bleads? to\b",
            r"\btherefore\b",
            r"\binstead of\b",
            r"\balthough\b",
        )

        normalized = sentence.casefold()
        return 0.15 if any(
            re.search(pattern, normalized)
            for pattern in patterns
        ) else 0.0

    @classmethod
    def _ranking_adjustment(
        cls,
        question: str,
        sentence: str,
    ) -> tuple[float, float]:
        overlap = cls._query_overlap_ratio(question, sentence)
        sentence_tokens = cls._content_tokens(sentence)
        query_tokens = cls._content_tokens(question)

        if not sentence_tokens:
            return 0.0, overlap

        novel_ratio = (
            len(sentence_tokens - query_tokens)
            / len(sentence_tokens)
        )

        # Penalize sentences that repeat query terms
        # while contributing very little new information.
        jaccard = cls._jaccard_similarity(question, sentence)
        penalty = (
            0.35
            if jaccard > 0.80 and novel_ratio < 0.25
            else 0.0
        )

        bonus = cls._information_bonus(sentence)
        return bonus - penalty, overlap

    @staticmethod
    def _normalize_aspect_queries(aspect_queries):
        normalized = []

        if not aspect_queries:
            return normalized

        for item in aspect_queries:
            if isinstance(item, dict):
                aspect = item.get("aspect")
                query = item.get("question") or item.get("query")

                if (
                    isinstance(aspect, str)
                    and aspect.strip()
                    and isinstance(query, str)
                    and query.strip()
                ):
                    normalized.append({
                        "aspect": aspect.strip(),
                        "query": query.strip(),
                    })

            elif isinstance(item, str):
                aspect = item.strip()

                if aspect:
                    normalized.append({
                        "aspect": aspect,
                        "query": aspect,
                    })

        return normalized

    def extract_candidates(
        self,
        question: str,
        context: str,
        aspect_queries: list[str] | None = None,
    ) -> list[dict[str, Any]]:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")

        if not isinstance(context, str) or not context.strip():
            return []

        sentences = self._split_sentences(context)

        if not sentences:
            return []

        if self.reranker is not None:
            ranked = self.reranker.rerank(question, sentences)
        else:
            ranked = [
                {
                    "index": idx,
                    "sentence": sentence,
                    "score": 1.0,
                }
                for idx, sentence in enumerate(sentences)
            ]

        ranked.sort(
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )

        # A6: absolute relevance filtering.
        # Do not force weak candidates merely to satisfy min_candidates.
        eligible = []

        for item in ranked:
            score = float(item.get("score", 0.0))

            if self.min_score is not None and score < self.min_score:
                continue

            eligible.append(item)

        if not eligible:
            return []

        normalized_aspect_queries = self._normalize_aspect_queries(
            aspect_queries
        )

        aspect_scores_by_query = {}

        for aspect_info in normalized_aspect_queries:
            query = aspect_info["query"]
            aspect = aspect_info["aspect"]

            if self.reranker is not None:
                aspect_ranked = self.reranker.rerank(query, sentences)

                score_map = {
                    item["sentence"]: float(item.get("score", 0.0))
                    for item in aspect_ranked
                }
            else:
                score_map = {
                    sentence: 1.0
                    for sentence in sentences
                }

            aspect_scores_by_query[aspect] = score_map

        selected = []

        for item in eligible:
            sentence = item.get("sentence")

            if not sentence:
                continue

            raw_score = float(item.get("score", 0.0))
            adjustment, query_overlap = self._ranking_adjustment(
                question,
                sentence,
            )
            ranking_score = raw_score + adjustment

            record = {
                "sentence": sentence,
                "score": ranking_score,
                "raw_score": raw_score,
                "ranking_score": ranking_score,
                "question_score": raw_score,
                "query_overlap": query_overlap,
                "index": item.get("index", len(selected)),
                "covered_aspects": [],
                "aspect_scores": {},
                "aspect_labels": {},
            }

            for aspect_info in normalized_aspect_queries:
                aspect_name = aspect_info["aspect"]

                aspect_score_map = aspect_scores_by_query.get(
                    aspect_name,
                    {},
                )

                score_for_sentence = aspect_score_map.get(
                    sentence,
                    0.0,
                )

                record["aspect_scores"][aspect_name] = score_for_sentence
                record["aspect_labels"][aspect_name] = aspect_name

                best_aspect_score = max(
                    aspect_score_map.values(),
                    default=0.0,
                )

                if (
                    score_for_sentence > 0.0
                    and score_for_sentence >= best_aspect_score
                ):
                    record["covered_aspects"].append(aspect_name)

            selected.append(record)

        if not selected:
            return []

        selected.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        best_score = selected[0]["score"]

        if self.score_gap_threshold > 0:
            candidates = [
                item
                for item in selected
                if item["score"] >= best_score - self.score_gap_threshold
            ]
        else:
            candidates = selected

        # min_candidates is now a target, not permission to include
        # candidates that failed the absolute relevance gate.
        return candidates[:self.max_candidates]

