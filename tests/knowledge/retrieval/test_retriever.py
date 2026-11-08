from unittest.mock import Mock

import pytest

from cognexa.knowledge.retrieval.retriever import Retriever


@pytest.fixture
def embedder():
    mock = Mock()
    mock.embed_text.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def vector_store():
    mock = Mock()
    mock.search.return_value = [
        {
            "id": "doc_1",
            "content": "Python is a programming language.",
            "metadata": {"source": "python.txt"},
            "distance": 0.05,
        }
    ]
    return mock


@pytest.fixture
def retriever(embedder, vector_store):
    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )


def test_retriever_initialization(retriever, embedder, vector_store):
    assert retriever.embedder is embedder
    assert retriever.vector_store is vector_store


def test_retrieve_returns_relevant_documents(
    retriever,
    embedder,
    vector_store,
):
    results = retriever.retrieve(
        query="What is Python?",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc_1"
    assert results[0]["content"] == "Python is a programming language."

    embedder.embed_text.assert_called_once_with("What is Python?")

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=3,
    )


def test_retrieve_uses_default_top_k(
    retriever,
    vector_store,
):
    retriever.retrieve("What is Python?")

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )


def test_retrieve_returns_empty_list_when_no_documents_found(
    retriever,
    vector_store,
):
    vector_store.search.return_value = []

    results = retriever.retrieve("Unknown topic")

    assert results == []


def test_retrieve_rejects_empty_query(retriever, embedder):
    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.retrieve("   ")

    embedder.embed_text.assert_not_called()


def test_retrieve_rejects_non_string_query(retriever, embedder):
    with pytest.raises(TypeError, match="query must be a string"):
        retriever.retrieve(123)

    embedder.embed_text.assert_not_called()


def test_retrieve_rejects_invalid_top_k(retriever, embedder):
    with pytest.raises(ValueError, match="greater than 0"):
        retriever.retrieve(
            query="What is Python?",
            top_k=0,
        )

    embedder.embed_text.assert_not_called()
