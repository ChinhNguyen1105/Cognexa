from unittest.mock import Mock

import pytest

from cognexa.knowledge.knowledge_base import KnowledgeBase


@pytest.fixture
def ingestion_pipeline():
    mock = Mock()
    mock.ingest.return_value = {
        "source": "C:/documents/python.txt",
        "file_type": "txt",
        "chunk_count": 2,
        "ids": ["doc_1", "doc_2"],
    }
    return mock


@pytest.fixture
def retriever():
    mock = Mock()
    mock.retrieve.return_value = [
        {
            "id": "doc_1",
            "content": "Python is a programming language.",
            "metadata": {"source": "python.txt"},
            "distance": 0.05,
        }
    ]
    return mock


@pytest.fixture
def knowledge_base(ingestion_pipeline, retriever):
    return KnowledgeBase(
        ingestion_pipeline=ingestion_pipeline,
        retriever=retriever,
    )


def test_knowledge_base_initialization(
    knowledge_base,
    ingestion_pipeline,
    retriever,
):
    assert knowledge_base.ingestion_pipeline is ingestion_pipeline
    assert knowledge_base.retriever is retriever


def test_add_document(knowledge_base, ingestion_pipeline):
    result = knowledge_base.add_document("python.txt")

    ingestion_pipeline.ingest.assert_called_once_with("python.txt")

    assert result["source"] == "C:/documents/python.txt"
    assert result["chunk_count"] == 2
    assert result["ids"] == ["doc_1", "doc_2"]


def test_add_document_returns_pipeline_result(
    knowledge_base,
    ingestion_pipeline,
):
    expected_result = {
        "source": "C:/documents/nestjs.pdf",
        "file_type": "pdf",
        "chunk_count": 5,
        "ids": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
    }

    ingestion_pipeline.ingest.return_value = expected_result

    result = knowledge_base.add_document("nestjs.pdf")

    assert result == expected_result


def test_search(knowledge_base, retriever):
    results = knowledge_base.search(
        query="What is Python?",
        top_k=3,
    )

    retriever.retrieve.assert_called_once_with(
        query="What is Python?",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc_1"


def test_search_uses_default_top_k(knowledge_base, retriever):
    knowledge_base.search("What is Python?")

    retriever.retrieve.assert_called_once_with(
        query="What is Python?",
        top_k=5,
    )


def test_search_returns_empty_list_when_no_results(
    knowledge_base,
    retriever,
):
    retriever.retrieve.return_value = []

    results = knowledge_base.search("Unknown topic")

    assert results == []


def test_add_document_propagates_error(
    knowledge_base,
    ingestion_pipeline,
):
    ingestion_pipeline.ingest.side_effect = FileNotFoundError(
        "Document not found"
    )

    with pytest.raises(FileNotFoundError, match="Document not found"):
        knowledge_base.add_document("missing.txt")


def test_search_propagates_error(knowledge_base, retriever):
    retriever.retrieve.side_effect = ValueError("Invalid query")

    with pytest.raises(ValueError, match="Invalid query"):
        knowledge_base.search("Some query")


