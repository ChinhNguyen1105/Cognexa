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
        min_score: float | None = None,
        neighbor_window: int = 1,
        neighbor_similarity_threshold: float = 0.15,
    ):
        if min_candidates <= 0:
            raise ValueError("min_candidates must be greater than zero")

        if max_candidates < min_candidates:
            raise ValueError(
                "max_candidates must be greater than or equal to min_candidates"
            )

        if score_gap_threshold < 0:
            raise ValueError("score_gap_threshold must be non-negative")

        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")

        if not 0.0 <= neighbor_similarity_threshold <= 1.0:
            raise ValueError(
                "neighbor_similarity_threshold must be between 0 and 1"
            )

        self.reranker = reranker
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.score_gap_threshold = score_gap_threshold
        self.min_score = min_score
        self.neighbor_window = neighbor_window
        self.neighbor_similarity_threshold = neighbor_similarity_threshold

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
            if jaccard > 0.60 and novel_ratio < 0.40
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

    @classmethod
    def _neighbor_similarity(cls, sentence_a: str, sentence_b: str) -> float:
        tokens_a = cls._content_tokens(sentence_a)
        tokens_b = cls._content_tokens(sentence_b)

        if not tokens_a or not tokens_b:
            return 0.0

        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    @classmethod
    def _apply_diversity_ranking(
        cls,
        candidates: list[dict[str, Any]],
        similarity_threshold: float = 0.60,
        redundancy_penalty: float = 0.35,
        novelty_bonus: float = 0.15,
        minimum_novel_ratio: float = 0.40,
    ) -> list[dict[str, Any]]:
        """
        Greedily rank candidates by relevance and newly contributed information.
        """
        remaining = list(candidates)
        selected = []
        selected_tokens = set()

        while remaining:
            best_candidate = None
            best_diversity_score = float("-inf")

            for candidate in remaining:
                sentence = candidate["sentence"]
                tokens = cls._content_tokens(sentence)

                if not tokens:
                    novel_ratio = 0.0
                else:
                    novel_ratio = (
                        len(tokens - selected_tokens) / len(tokens)
                    )

                max_similarity = max(
                    (
                        cls._neighbor_similarity(
                            sentence,
                            selected_item["sentence"],
                        )
                        for selected_item in selected
                    ),
                    default=0.0,
                )

                penalty = (
                    redundancy_penalty
                    if (
                        max_similarity > similarity_threshold
                        and novel_ratio < minimum_novel_ratio
                    )
                    else 0.0
                )

                base_score = float(
                    candidate.get(
                        "ranking_score",
                        candidate.get("score", 0.0),
                    )
                )

                diversity_score = (
                    base_score
                    - penalty
                    + novelty_bonus * novel_ratio
                )

                if diversity_score > best_diversity_score:
                    best_candidate = candidate
                    best_diversity_score = diversity_score

            if best_candidate is None:
                break

            best_candidate["diversity_score"] = best_diversity_score
            selected.append(best_candidate)

            selected_tokens.update(
                cls._content_tokens(best_candidate["sentence"])
            )

            remaining.remove(best_candidate)

        return selected

    def _expand_neighbors(
        self,
        candidates: list[dict[str, Any]],
        sentences: list[str],
        ranked: list[dict[str, Any]],
        question: str,
        normalized_aspect_queries: list[dict[str, str]],
        aspect_scores_by_query: dict,
    ) -> list[dict[str, Any]]:
        if self.neighbor_window == 0 or not candidates:
            return candidates

        max_raw_score = max(
            (
                float(item.get("raw_score", item.get("score", 0.0)))
                for item in candidates
            ),
            default=0.0,
        )

        ranked_by_index = {
            int(item["index"]): item
            for item in ranked
            if "index" in item
        }

        selected_indices = {
            int(item["index"])
            for item in candidates
            if item.get("index") is not None
        }

        neighbors = []

        for seed in candidates:
            seed_index = int(seed["index"])

            for offset in range(
                -self.neighbor_window,
                self.neighbor_window + 1,
            ):
                if offset == 0:
                    continue

                neighbor_index = seed_index + offset

                if not 0 <= neighbor_index < len(sentences):
                    continue

                if neighbor_index in selected_indices:
                    continue

                sentence = sentences[neighbor_index]
                similarity = self._neighbor_similarity(
                    seed["sentence"],
                    sentence,
                )

                if similarity < self.neighbor_similarity_threshold:
                    continue

                ranked_item = ranked_by_index.get(neighbor_index, {})
                raw_score = float(ranked_item.get("score", 0.0))

                if self.score_gap_threshold > 0 and raw_score < max_raw_score - self.score_gap_threshold:
                    continue

                adjustment, query_overlap = self._ranking_adjustment(
                    question,
                    sentence,
                )

                aspect_scores = {}
                aspect_labels = {}
                covered_aspects = []

                for aspect_info in normalized_aspect_queries:
                    aspect_name = aspect_info["aspect"]
                    score_map = aspect_scores_by_query.get(
                        aspect_name,
                        {},
                    )

                    aspect_score = score_map.get(sentence, 0.0)

                    aspect_scores[aspect_name] = aspect_score
                    aspect_labels[aspect_name] = aspect_name

                    best_score = max(score_map.values(), default=0.0)

                    if score_map and aspect_score == best_score:
                        covered_aspects.append(aspect_name)

                neighbors.append({
                    "sentence": sentence,
                    "score": raw_score + adjustment,
                    "raw_score": raw_score,
                    "ranking_score": raw_score + adjustment,
                    "question_score": raw_score,
                    "query_overlap": query_overlap,
                    "index": neighbor_index,
                    "covered_aspects": covered_aspects,
                    "aspect_scores": aspect_scores,
                    "aspect_labels": aspect_labels,
                    "is_neighbor": True,
                    "seed_index": seed_index,
                })

                selected_indices.add(neighbor_index)

        expanded = candidates + neighbors
        expanded.sort(
            key=lambda item: item.get(
                "diversity_score",
                item["score"],
            ),
            reverse=True,
        )

        return expanded[:self.max_candidates]

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
        # CrossEncoder logits are not calibrated probabilities; a negative
        # score can still be the strongest candidate in a given query. Only
        # enforce a floor when a project explicitly configures one.
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
                    score_for_sentence >= best_aspect_score
                    and score_for_sentence == best_aspect_score
                    and best_aspect_score != 0.0
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
        best_raw_score = selected[0]["raw_score"]

        if self.score_gap_threshold > 0:
            gap_candidates = [
                item
                for item in selected
                if item["raw_score"] >= best_raw_score - self.score_gap_threshold
            ]
            if len(gap_candidates) >= self.min_candidates:
                candidates = gap_candidates
            elif normalized_aspect_queries:
                preserved = list(gap_candidates)
                for item in selected:
                    if item in gap_candidates:
                        continue

                    if any(
                        self._neighbor_similarity(
                            item["sentence"],
                            kept["sentence"],
                        ) > 0.55
                        for kept in preserved
                    ):
                        continue

                    preserved.append(item)

                if len(preserved) < self.min_candidates:
                    preserved = selected[: self.max_candidates]

                candidates = preserved[: self.max_candidates]
            else:
                candidates = gap_candidates
        else:
            candidates = selected

        # Prefer candidates that contribute new information.
        diverse_candidates = self._apply_diversity_ranking(candidates)

        # Keep adjacent context that is highly topical and semantically continuous.
        ranked_by_index = {
            int(item["index"]): item
            for item in ranked
            if "index" in item
        }
        extended = list(diverse_candidates)
        seen_indices = {int(item.get("index", -1)) for item in extended if isinstance(item.get("index"), int)}
        max_candidate_raw = max(
            (float(item.get("raw_score", item.get("score", 0.0))) for item in extended),
            default=0.0,
        )
        for item in list(diverse_candidates):
            idx = int(item.get("index", -1))
            if idx < 0 or idx + 1 >= len(sentences):
                continue

            next_idx = idx + 1
            if next_idx in seen_indices:
                continue

            next_sentence = sentences[next_idx]
            next_overlap = self._query_overlap_ratio(question, next_sentence)
            next_tokens = self._content_tokens(next_sentence)
            if next_overlap < 0.20 or not next_tokens:
                continue

            next_raw_score = float(
                ranked_by_index.get(next_idx, {}).get("score", item.get("raw_score", item.get("score", 0.0)))
            )
            if self.score_gap_threshold > 0 and next_raw_score < max_candidate_raw - self.score_gap_threshold:
                continue

            leader = self._content_tokens(item["sentence"])
            if next_tokens.issubset(leader) or leader.issubset(next_tokens):
                continue

            extended.append({
                "sentence": next_sentence,
                "score": item.get("score", 0.0) * 0.85 + next_overlap * 0.15,
                "raw_score": next_raw_score,
                "ranking_score": item.get("score", 0.0) * 0.85 + next_overlap * 0.15,
                "question_score": item.get("score", 0.0),
                "query_overlap": next_overlap,
                "index": next_idx,
                "covered_aspects": [],
                "aspect_scores": {},
                "aspect_labels": {},
                "is_neighbor": True,
            })
            seen_indices.add(next_idx)

        # Expand selected evidence with relevant neighboring sentences.
        return self._expand_neighbors(
            candidates=extended,
            sentences=sentences,
            ranked=ranked,
            question=question,
            normalized_aspect_queries=normalized_aspect_queries,
            aspect_scores_by_query=aspect_scores_by_query,
        )

