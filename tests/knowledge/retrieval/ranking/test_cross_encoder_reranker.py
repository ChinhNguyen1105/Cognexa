import pytest

from cognexa.knowledge.retrieval.contracts import (
    DocumentChunk,
    FusedCandidate,
    RetrievalMatch,
    RetrievalQuery,
    RetrieverType,
)
from cognexa.knowledge.retrieval.ranking import (
    CrossEncoderCandidateReranker,
    CrossEncoderReranker,
)


class FakeCrossEncoderModel:
    """Fake model that returns predefined relevance scores."""

    def __init__(self, scores):
        self.scores = scores
        self.received_pairs = None

    def predict(
        self,
        pairs,
        batch_size=16,
        show_progress_bar=False,
    ):
        self.received_pairs = pairs
        return self.scores


@pytest.fixture
def query():
    return RetrievalQuery(
        query_id="query_001",
        text="What is machine learning?",
    )


@pytest.fixture
def candidates():
    contents = [
        "Python is a programming language.",
        "Machine learning allows systems to learn from data.",
        "The weather is sunny today.",
    ]

    result = []

    for index, content in enumerate(contents, start=1):
        chunk = DocumentChunk(
            chunk_id=f"chunk_{index}",
            document_id=f"doc_{index}",
            content=content,
        )

        match = RetrievalMatch(
            query_id="query_001",
            retriever=RetrieverType.DENSE,
            rank=index,
            raw_score=0.1 * index,
            score_type="distance",
        )

        result.append(
            FusedCandidate(
                chunk=chunk,
                rrf_score=1.0 / index,
                matches=[match],
                fusion_rank=index,
            )
        )

    return result


@pytest.fixture
def fake_reranker():
    scorer = CrossEncoderReranker()

    fake_model = FakeCrossEncoderModel(
        scores=[0.2, 0.95, 0.1]
    )

    # Inject fake model to avoid downloading a real model.
    scorer.model = fake_model

    adapter = CrossEncoderCandidateReranker(
        scorer=scorer
    )

    return adapter, fake_model


def test_rerank_returns_ranked_candidates(
    query,
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    results = adapter.rerank(query, candidates)

    assert len(results) == 3
    assert all(
        result.query_id == query.query_id
        for result in results
    )
    assert all(
        isinstance(result.candidate, FusedCandidate)
        for result in results
    )


def test_rerank_sorts_by_score_descending(
    query,
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    results = adapter.rerank(query, candidates)

    assert [
        result.rerank_score for result in results
    ] == [0.95, 0.2, 0.1]

    assert [
        result.candidate.chunk.chunk_id
        for result in results
    ] == ["chunk_2", "chunk_1", "chunk_3"]


def test_rerank_assigns_sequential_ranks(
    query,
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    results = adapter.rerank(query, candidates)

    assert [
        result.rerank_rank for result in results
    ] == [1, 2, 3]


def test_rerank_preserves_original_candidates(
    query,
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    results = adapter.rerank(query, candidates)

    ranked_by_id = {
        result.candidate.chunk.chunk_id: result.candidate
        for result in results
    }

    for candidate in candidates:
        assert (
            ranked_by_id[candidate.chunk.chunk_id]
            is candidate
        )


def test_rerank_passes_query_and_chunk_content_to_model(
    query,
    candidates,
    fake_reranker,
):
    adapter, fake_model = fake_reranker

    adapter.rerank(query, candidates)

    assert fake_model.received_pairs == [
        (
            query.text,
            candidate.chunk.content,
        )
        for candidate in candidates
    ]


def test_rerank_respects_top_k(
    query,
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    results = adapter.rerank(
        query,
        candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert [result.rerank_rank for result in results] == [1, 2]


def test_rerank_returns_empty_list_for_empty_candidates(
    query,
    fake_reranker,
):
    adapter, fake_model = fake_reranker

    results = adapter.rerank(query, [])

    assert results == []
    assert fake_model.received_pairs is None


def test_rerank_rejects_invalid_top_k(
    query,
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    with pytest.raises(ValueError):
        adapter.rerank(query, candidates, top_k=0)

    with pytest.raises(TypeError):
        adapter.rerank(query, candidates, top_k="2")


def test_rerank_rejects_invalid_query(
    candidates,
    fake_reranker,
):
    adapter, _ = fake_reranker

    with pytest.raises(TypeError):
        adapter.rerank("machine learning", candidates)


def test_rerank_rejects_invalid_candidates(
    query,
    fake_reranker,
):
    adapter, _ = fake_reranker

    with pytest.raises(TypeError):
        adapter.rerank(query, ["not a fused candidate"])


def test_reranker_does_not_load_model_during_initialization():
    scorer = CrossEncoderReranker()

    adapter = CrossEncoderCandidateReranker(
        scorer=scorer
    )

    assert adapter.scorer.model is None
