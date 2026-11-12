from dataclasses import dataclass, field
from collections.abc import Sequence

from ..contracts import (
    FusedCandidate,
    RetrievalMatch,
    RetrievalRun,
)


@dataclass
class _CandidateAccumulator:
    chunk: object
    rrf_score: float = 0.0
    matches: list[RetrievalMatch] = field(default_factory=list)


class ReciprocalRankFusion:
    """
    Merge ranked retrieval runs using Reciprocal Rank Fusion.

    Each RetrievalRun contributes independently to the final score.
    Documents are deduplicated using chunk_id.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self.rrf_k = rrf_k

    def fuse(
        self,
        runs: Sequence[RetrievalRun],
        top_k: int | None = None,
    ) -> list[FusedCandidate]:
        """
        Merge multiple retrieval runs into a ranked candidate list.

        Args:
            runs: Dense/BM25 retrieval runs.
            top_k: Maximum number of fused candidates.
                  None means return all candidates.

        Returns:
            Candidates sorted by descending RRF score.
        """
        if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
            raise TypeError("runs must be a sequence of RetrievalRun")

        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError("top_k must be an integer or None")

            if top_k <= 0:
                raise ValueError("top_k must be greater than 0")

        accumulators: dict[str, _CandidateAccumulator] = {}

        for run in runs:
            if not isinstance(run, RetrievalRun):
                raise TypeError("Every item in runs must be a RetrievalRun")

            # A chunk should contribute at most once per retrieval run.
            best_hits = {}

            for hit in run.hits:
                if isinstance(hit.rank, bool) or not isinstance(hit.rank, int):
                    raise TypeError("Retrieval hit rank must be an integer")

                if hit.rank <= 0:
                    raise ValueError("Retrieval hit rank must be greater than 0")

                if hit.query_id != run.query_id:
                    raise ValueError(
                        "Retrieval hit query_id must match its RetrievalRun"
                    )

                chunk_id = hit.chunk.chunk_id

                current = best_hits.get(chunk_id)

                if current is None or hit.rank < current.rank:
                    best_hits[chunk_id] = hit

            for chunk_id, hit in best_hits.items():
                accumulator = accumulators.get(chunk_id)

                if accumulator is None:
                    accumulator = _CandidateAccumulator(
                        chunk=hit.chunk,
                    )
                    accumulators[chunk_id] = accumulator

                accumulator.rrf_score += 1.0 / (
                    self.rrf_k + hit.rank
                )

                accumulator.matches.append(
                    RetrievalMatch(
                        query_id=hit.query_id,
                        retriever=hit.retriever,
                        rank=hit.rank,
                        raw_score=hit.raw_score,
                        score_type=hit.score_type,
                    )
                )

        ranked_candidates = sorted(
            accumulators.values(),
            key=lambda item: item.rrf_score,
            reverse=True,
        )

        if top_k is not None:
            ranked_candidates = ranked_candidates[:top_k]

        return [
            FusedCandidate(
                chunk=item.chunk,
                rrf_score=item.rrf_score,
                matches=list(item.matches),
                fusion_rank=rank,
            )
            for rank, item in enumerate(ranked_candidates, start=1)
        ]

