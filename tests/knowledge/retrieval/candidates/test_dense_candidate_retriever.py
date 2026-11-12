import pytest

from cognexa.knowledge.retrieval.candidates.dense_retriever import (
    DenseRetriever,
)
from cognexa.knowledge.retrieval.contracts import (
    RetrievalQuery,
    RetrieverType,
)


class FakeEmbedder:
    def __init__(self):
        self.received_text = None

    def embed_text(self, text):
        self.received_text = text
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.received_embedding = None
        self.received_top_k = None

    def search(self, query_embedding, top_k):
        self.received_embedding = query_embedding
        self.received_top_k = top_k
        return self.results


def make_query():
    return RetrievalQuery(
        query_id="q1",
        text="What is Python?",
    )


def make_vector_result():
    return {
        "id": "chunk-001",
        "document": "Python is a programming language.",
        "metadata": {
            "document_id": "python-doc",
            "source": "python.txt",
            "parent_id": "parent-001",
            "chunk_index": 0,
        },
        "distance": 0.15,
    }


def make_retriever(results=None):
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(
        results if results is not None else [make_vector_result()]
    )

    return (
        DenseRetriever(embedder, vector_store),
        embedder,
        vector_store,
    )


def test_dense_retriever_returns_retrieval_run():
    retriever, _, _ = make_retriever()

    result = retriever.retrieve(make_query(), top_k=3)

    assert result.query_id == "q1"
    assert result.retriever == RetrieverType.DENSE
    assert len(result.hits) == 1


def test_dense_retriever_uses_embedder():
    retriever, embedder, _ = make_retriever()

    retriever.retrieve(make_query())

    assert embedder.received_text == "What is Python?"


def test_dense_retriever_passes_embedding_to_vector_store():
    retriever, _, vector_store = make_retriever()

    retriever.retrieve(make_query(), top_k=7)

    assert vector_store.received_embedding == [0.1, 0.2, 0.3]
    assert vector_store.received_top_k == 7


def test_dense_retriever_maps_document_chunk():
    retriever, _, _ = make_retriever()

    result = retriever.retrieve(make_query())
    chunk = result.hits[0].chunk

    assert chunk.chunk_id == "chunk-001"
    assert chunk.document_id == "python-doc"
    assert chunk.parent_chunk_id == "parent-001"
    assert chunk.position == 0


def test_dense_retriever_preserves_distance_score():
    retriever, _, _ = make_retriever()

    result = retriever.retrieve(make_query())
    hit = result.hits[0]

    assert hit.raw_score == 0.15
    assert hit.score_type == "distance"
    assert hit.rank == 1


def test_dense_retriever_returns_empty_run():
    retriever, _, _ = make_retriever(results=[])

    result = retriever.retrieve(make_query())

    assert result.hits == []
    assert result.retriever == RetrieverType.DENSE


def test_dense_retriever_rejects_invalid_query():
    retriever, _, _ = make_retriever()

    with pytest.raises(TypeError):
        retriever.retrieve("What is Python?")


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_dense_retriever_rejects_invalid_top_k(top_k):
    retriever, _, _ = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(make_query(), top_k=top_k)


def test_dense_retriever_rejects_missing_chunk_id():
    result = {
        "document": "Python is a programming language.",
        "metadata": {},
        "distance": 0.15,
    }

    retriever, _, _ = make_retriever([result])

    with pytest.raises(ValueError):
        retriever.retrieve(make_query())


def test_dense_retriever_rejects_missing_score():
    result = {
        "id": "chunk-001",
        "document": "Python is a programming language.",
        "metadata": {},
    }

    retriever, _, _ = make_retriever([result])

    with pytest.raises(ValueError):
        retriever.retrieve(make_query())