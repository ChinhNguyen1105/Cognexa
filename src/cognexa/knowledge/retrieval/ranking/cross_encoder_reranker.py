from collections.abc import Sequence

from cognexa.knowledge.retrieval.contracts import (
    FusedCandidate,
    RankedCandidate,
    RetrievalQuery,
)

from .reranker import CrossEncoderReranker


class CrossEncoderCandidateReranker:
    """
    Structured Cross-Encoder reranker.

    Implements the RerankerProtocol contract:
        rerank(query, candidates, top_k) -> list[RankedCandidate]

    Reuses the legacy CrossEncoderReranker for model loading
    and relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: str | None = None,
        batch_size: int = 16,
        scorer: CrossEncoderReranker | None = None,
    ) -> None:
        if scorer is not None and not isinstance(
            scorer, CrossEncoderReranker
        ):
            raise TypeError(
                "scorer must be a CrossEncoderReranker instance"
            )

        self.scorer = scorer or CrossEncoderReranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[FusedCandidate],
        top_k: int | None = None,
    ) -> list[RankedCandidate]:
        """
        Rerank fused candidates using Cross-Encoder relevance scores.

        The original FusedCandidate is preserved inside each
        RankedCandidate.
        """
        if not isinstance(query, RetrievalQuery):
            raise TypeError("query must be a RetrievalQuery")

        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Sequence
        ):
            raise TypeError(
                "candidates must be a sequence of FusedCandidate"
            )

        if any(
            not isinstance(candidate, FusedCandidate)
            for candidate in candidates
        ):
            raise TypeError(
                "Every candidate must be a FusedCandidate"
            )

        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError("top_k must be an integer or None")

            if top_k <= 0:
                raise ValueError("top_k must be greater than 0")

        if not candidates:
            return []

        candidate_texts = [
            candidate.chunk.content
            for candidate in candidates
        ]

        scored_results = self.scorer.rerank(
            query=query.text,
            candidates=candidate_texts,
        )

        ranked_pairs = [
            (
                candidates[item["index"]],
                float(item["score"]),
            )
            for item in scored_results
        ]

        ranked_pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        if top_k is not None:
            ranked_pairs = ranked_pairs[:top_k]

        return [
            RankedCandidate(
                query_id=query.query_id,
                candidate=candidate,
                rerank_score=score,
                rerank_rank=rank,
            )
            for rank, (candidate, score) in enumerate(
                ranked_pairs,
                start=1,
            )
        ]
