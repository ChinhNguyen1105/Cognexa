from cognexa.knowledge.reranking.reranker import CrossEncoderReranker
import re


class AnswerExtractor:
    """Rank context sentences against the original question and aspects."""

    def __init__(
        self,
        reranker: CrossEncoderReranker,
        min_candidates: int = 2,
        max_candidates: int = 24,
        score_gap_threshold: float = 3.0,
    ):
        if min_candidates <= 0:
            raise ValueError("min_candidates must be greater than zero.")
        if max_candidates < min_candidates:
            raise ValueError("max_candidates must be >= min_candidates.")
        if score_gap_threshold < 0:
            raise ValueError("score_gap_threshold must be >= 0.")

        self.reranker = reranker
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.score_gap_threshold = score_gap_threshold

    def extract_candidates(
        self,
        question: str,
        context: str,
        aspect_queries: list[str | dict] | None = None,
    ) -> list[dict]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(context, str) or not context.strip():
            return []

        sentences = self._split_sentences(context)

        if not sentences:
            return []

        normalized_question = question.strip()

        queries = [
            {
                "aspect": "original",
                "question": normalized_question,
            }
        ]

        seen_queries = {normalized_question.casefold()}
        seen_aspects = {"original"}

        for index, item in enumerate(aspect_queries or [], start=1):
            if isinstance(item, str):
                aspect = f"aspect_{index}"
                query = item.strip()

            elif isinstance(item, dict):
                aspect = item.get("aspect")
                query = item.get("question")

                if not isinstance(aspect, str) or not isinstance(query, str):
                    continue

                aspect = aspect.strip().casefold()
                query = query.strip()

            else:
                continue

            if not aspect or not query:
                continue

            normalized_query = query.casefold()

            if normalized_query in seen_queries:
                continue

            if aspect in seen_aspects:
                aspect = f"{aspect}_{index}"

            queries.append({
                "aspect": aspect,
                "question": query,
            })

            seen_queries.add(normalized_query)
            seen_aspects.add(aspect)

        by_sentence = {}

        for query_info in queries:
            aspect = query_info["aspect"]
            query = query_info["question"]

            ranked = self.reranker.rerank(
                query=query,
                candidates=sentences,
            ) or []

            for item in ranked:
                if not isinstance(item, dict):
                    continue

                index = item.get("index")

                if (
                    not isinstance(index, int)
                    or not 0 <= index < len(sentences)
                ):
                    continue

                sentence = item.get("sentence", sentences[index])
                score = item.get("score")

                if not isinstance(sentence, str):
                    continue

                if not isinstance(score, (int, float)):
                    continue

                entry = by_sentence.setdefault(
                    index,
                    {
                        "index": index,
                        "sentence": sentence,
                        "score": float("-inf"),
                        "question_score": None,
                        "aspect_scores": {},
                        "aspect_labels": {},
                        "antecedents": [],
                    },
                )

                if aspect == "original":
                    entry["question_score"] = float(score)
                    entry["score"] = float(score)
                else:
                    # Keep query text as the score key for backward compatibility.
                    entry["aspect_scores"][query] = float(score)
                    entry["aspect_labels"][query] = aspect

        ranked_candidates = sorted(
            by_sentence.values(),
            key=lambda item: (
                item["question_score"]
                if item["question_score"] is not None
                else float("-inf")
            ),
            reverse=True,
        )

        pool = self._build_candidate_pool(ranked_candidates)

        # Preserve complementary candidates retrieved by aspect queries.
        for query_info in queries[1:]:
            query = query_info["question"]

            aspect_ranked = sorted(
                (
                    item
                    for item in by_sentence.values()
                    if query in item["aspect_scores"]
                ),
                key=lambda item: item["aspect_scores"][query],
                reverse=True,
            )

            for item in aspect_ranked[:self.min_candidates]:
                if all(
                    existing["index"] != item["index"]
                    for existing in pool
                ):
                    pool.append(item)

        pool.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return pool[:self.max_candidates]

    def _build_candidate_pool(
        self,
        ranked_candidates: list[dict],
    ) -> list[dict]:
        if not ranked_candidates:
            return []

        best_score = ranked_candidates[0]["score"]
        selected = []

        for candidate in ranked_candidates:
            if len(selected) >= self.max_candidates:
                break

            if (
                len(selected) < self.min_candidates
                or candidate["score"]
                >= best_score - self.score_gap_threshold
            ):
                selected.append(candidate)
            else:
                break

        return selected

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        raw = re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip(),
        )

        result = []
        seen = set()

        for sentence in raw:
            sentence = sentence.strip()

            if not sentence or sentence.startswith("[Document "):
                continue

            normalized = sentence.casefold()

            if normalized not in seen:
                result.append(sentence)
                seen.add(normalized)

        return result
