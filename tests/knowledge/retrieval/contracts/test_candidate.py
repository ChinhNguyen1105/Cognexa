import pytest

from cognexa.knowledge.retrieval.contracts import (
    DocumentChunk,
    FusedCandidate,
    RankedCandidate,
    RetrievalHit,
    RetrievalMatch,
    RetrievalRun,
    RetrieverType,
)


def make_chunk():
    return DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        content="Python is a programming language.",
    )


def make_hit(retriever=RetrieverType.DENSE, query_id="q1"):
    return RetrievalHit(
        chunk=make_chunk(),
        query_id=query_id,
        retriever=retriever,
        rank=1,
        raw_score=0.89,
        score_type="similarity",
    )


def make_fused_candidate():
    match = RetrievalMatch(
        query_id="q1",
        retriever=RetrieverType.DENSE,
        rank=1,
        raw_score=0.89,
        score_type="similarity",
    )

    return FusedCandidate(
        chunk=make_chunk(),
        rrf_score=1 / 61,
        matches=[match],
        fusion_rank=1,
    )


def test_retrieval_hit_creation():
    hit = make_hit()

    assert hit.retriever == RetrieverType.DENSE
    assert hit.rank == 1
    assert hit.raw_score == 0.89


def test_retrieval_run_creation():
    hit = make_hit()

    run = RetrievalRun(
        query_id="q1",
        retriever=RetrieverType.DENSE,
        hits=[hit],
    )

    assert len(run.hits) == 1


def test_retrieval_run_rejects_mismatched_query():
    hit = make_hit(query_id="q2")

    with pytest.raises(ValueError):
        RetrievalRun(
            query_id="q1",
            retriever=RetrieverType.DENSE,
            hits=[hit],
        )


def test_retrieval_run_rejects_duplicate_chunks():
    hit1 = make_hit()
    hit2 = make_hit()

    with pytest.raises(ValueError):
        RetrievalRun(
            query_id="q1",
            retriever=RetrieverType.DENSE,
            hits=[hit1, hit2],
        )


def test_retrieval_run_rejects_mismatched_retriever():
    hit = make_hit(retriever=RetrieverType.DENSE)

    with pytest.raises(ValueError):
        RetrievalRun(
            query_id="q1",
            retriever=RetrieverType.BM25,
            hits=[hit],
        )


def test_fused_candidate_creation():
    candidate = make_fused_candidate()

    assert candidate.fusion_rank == 1
    assert candidate.rrf_score > 0
    assert len(candidate.matches) == 1


def test_fused_candidate_requires_matches():
    with pytest.raises(ValueError):
        FusedCandidate(
            chunk=make_chunk(),
            rrf_score=0.01,
            matches=[],
            fusion_rank=1,
        )


def test_ranked_candidate_creation():
    candidate = RankedCandidate(
        query_id="q1",
        candidate=make_fused_candidate(),
        rerank_score=8.5,
        rerank_rank=1,
    )

    assert candidate.query_id == "q1"
    assert candidate.rerank_score == 8.5
    assert candidate.rerank_rank == 1


def test_ranked_candidate_rejects_invalid_rank():
    with pytest.raises(ValueError):
        RankedCandidate(
            query_id="q1",
            candidate=make_fused_candidate(),
            rerank_score=8.5,
            rerank_rank=0,
        )