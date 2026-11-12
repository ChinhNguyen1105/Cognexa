import pytest

from cognexa.knowledge.retrieval.contracts import (
    DocumentChunk,
    RetrievalHit,
    RetrievalRun,
    RetrieverType,
)
from cognexa.knowledge.retrieval.fusion.rrf_fusion import (
    ReciprocalRankFusion,
)


def make_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"Content of {chunk_id}",
    )


def make_run(
    query_id: str,
    retriever: RetrieverType,
    ranked_chunks: list[tuple[DocumentChunk, int]],
) -> RetrievalRun:
    hits = [
        RetrievalHit(
            chunk=chunk,
            query_id=query_id,
            retriever=retriever,
            rank=rank,
            raw_score=0.9 / rank,
            score_type=(
                "distance"
                if retriever == RetrieverType.DENSE
                else "bm25"
            ),
        )
        for chunk, rank in ranked_chunks
    ]

    return RetrievalRun(
        query_id=query_id,
        retriever=retriever,
        hits=hits,
    )


class TestReciprocalRankFusion:

    def test_fuse_single_retrieval_run(self):
        chunk_a = make_chunk("A")
        chunk_b = make_chunk("B")

        run = make_run(
            "q1",
            RetrieverType.DENSE,
            [(chunk_a, 1), (chunk_b, 2)],
        )

        results = ReciprocalRankFusion().fuse([run])

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "A"
        assert results[1].chunk.chunk_id == "B"

        assert results[0].rrf_score == pytest.approx(1 / 61)
        assert results[1].rrf_score == pytest.approx(1 / 62)

        assert results[0].fusion_rank == 1
        assert results[1].fusion_rank == 2

    def test_fuse_dense_and_bm25_runs(self):
        chunk_a = make_chunk("A")
        chunk_b = make_chunk("B")
        chunk_c = make_chunk("C")

        dense_run = make_run(
            "q1",
            RetrieverType.DENSE,
            [(chunk_a, 1), (chunk_b, 2)],
        )

        bm25_run = make_run(
            "q1",
            RetrieverType.BM25,
            [(chunk_b, 1), (chunk_c, 2)],
        )

        results = ReciprocalRankFusion().fuse(
            [dense_run, bm25_run]
        )

        assert [item.chunk.chunk_id for item in results] == [
            "B",
            "A",
            "C",
        ]

        expected_b_score = (1 / 62) + (1 / 61)

        assert results[0].rrf_score == pytest.approx(
            expected_b_score
        )
        assert results[1].rrf_score == pytest.approx(1 / 61)
        assert results[2].rrf_score == pytest.approx(1 / 62)

    def test_duplicate_chunks_are_merged_across_runs(self):
        chunk_a = make_chunk("A")

        dense_run = make_run(
            "q1",
            RetrieverType.DENSE,
            [(chunk_a, 1)],
        )

        bm25_run = make_run(
            "q1",
            RetrieverType.BM25,
            [(chunk_a, 2)],
        )

        results = ReciprocalRankFusion().fuse(
            [dense_run, bm25_run]
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "A"
        assert results[0].rrf_score == pytest.approx(
            (1 / 61) + (1 / 62)
        )
        assert len(results[0].matches) == 2

    def test_matches_preserve_retrieval_metadata(self):
        chunk_a = make_chunk("A")

        dense_run = make_run(
            "q1",
            RetrieverType.DENSE,
            [(chunk_a, 1)],
        )

        bm25_run = make_run(
            "q1",
            RetrieverType.BM25,
            [(chunk_a, 3)],
        )

        result = ReciprocalRankFusion().fuse(
            [dense_run, bm25_run]
        )[0]

        assert len(result.matches) == 2

        assert result.matches[0].retriever == RetrieverType.DENSE
        assert result.matches[0].rank == 1

        assert result.matches[1].retriever == RetrieverType.BM25
        assert result.matches[1].rank == 3

    def test_fused_candidate_matches_is_a_list(self):
        run = make_run(
            "q1",
            RetrieverType.DENSE,
            [(make_chunk("A"), 1)],
        )

        result = ReciprocalRankFusion().fuse([run])[0]

        assert isinstance(result.matches, list)

    def test_top_k_limits_results(self):
        run = make_run(
            "q1",
            RetrieverType.DENSE,
            [
                (make_chunk("A"), 1),
                (make_chunk("B"), 2),
                (make_chunk("C"), 3),
            ],
        )

        results = ReciprocalRankFusion().fuse(
            [run],
            top_k=2,
        )

        assert len(results) == 2
        assert [item.chunk.chunk_id for item in results] == [
            "A",
            "B",
        ]

    def test_empty_runs_return_empty_list(self):
        assert ReciprocalRankFusion().fuse([]) == []

        empty_run = RetrievalRun(
            query_id="q1",
            retriever=RetrieverType.DENSE,
        )

        assert ReciprocalRankFusion().fuse([empty_run]) == []

    def test_invalid_rrf_k_rejected(self):
        with pytest.raises(ValueError, match="greater than 0"):
            ReciprocalRankFusion(rrf_k=0)

        with pytest.raises(TypeError, match="integer"):
            ReciprocalRankFusion(rrf_k=1.5)

        with pytest.raises(TypeError, match="integer"):
            ReciprocalRankFusion(rrf_k=True)

    def test_invalid_top_k_rejected(self):
        run = make_run(
            "q1",
            RetrieverType.DENSE,
            [(make_chunk("A"), 1)],
        )

        with pytest.raises(ValueError, match="greater than 0"):
            ReciprocalRankFusion().fuse([run], top_k=0)

        with pytest.raises(TypeError, match="integer"):
            ReciprocalRankFusion().fuse([run], top_k=1.5)

    def test_invalid_runs_rejected(self):
        fusion = ReciprocalRankFusion()

        with pytest.raises(TypeError, match="sequence"):
            fusion.fuse(None)

        with pytest.raises(TypeError, match="RetrievalRun"):
            fusion.fuse([object()])
