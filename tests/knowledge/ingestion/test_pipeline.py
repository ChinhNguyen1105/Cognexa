from unittest.mock import Mock, call

import pytest

from cognexa.knowledge.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def loader():
    mock = Mock()
    mock.load.return_value = {
        "content": "Python is a programming language.",
        "metadata": {
            "source": "C:/documents/python.txt",
            "file_name": "python.txt",
            "file_type": "txt",
        },
    }
    return mock


@pytest.fixture
def chunker():
    mock = Mock()
    mock.chunk_text.return_value = [
        "Python is a programming language."
    ]
    return mock


@pytest.fixture
def embedder():
    mock = Mock()
    mock.embed_texts.return_value = [
        [0.1, 0.2, 0.3]
    ]
    return mock


@pytest.fixture
def vector_store():
    return Mock()


@pytest.fixture
def pipeline(loader, chunker, embedder, vector_store):
    return IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )


def test_pipeline_initialization(
    pipeline,
    loader,
    chunker,
    embedder,
    vector_store,
):
    assert pipeline.loader is loader
    assert pipeline.chunker is chunker
    assert pipeline.embedder is embedder
    assert pipeline.vector_store is vector_store
    assert pipeline.parent_chunker.chunk_size == 1500
    assert pipeline.parent_chunker.chunk_overlap == 150


def test_ingest_document_returns_correct_result(
    pipeline,
    vector_store,
):
    result = pipeline.ingest("python.txt")

    assert result["source"] == "C:/documents/python.txt"
    assert result["file_type"] == "txt"
    assert result["chunk_count"] == 1
    assert result["parent_count"] == 1
    assert result["child_count"] == 1

    assert len(result["parent_ids"]) == 1
    assert len(result["child_ids"]) == 1

    vector_store.add_documents.assert_called_once()
    vector_store.add_parent_documents.assert_called_once()


def test_ingest_calls_components_in_correct_order(
    pipeline,
    loader,
    chunker,
    embedder,
    vector_store,
):
    pipeline.ingest("python.txt")

    loader.load.assert_called_once_with("python.txt")

    chunker.chunk_text.assert_called_once_with(
        "Python is a programming language."
    )

    assert embedder.embed_texts.call_args_list == [
        call(["Python is a programming language."]),
        call(["Python is a programming language."]),
    ]

    vector_store.add_documents.assert_called_once()
    vector_store.add_parent_documents.assert_called_once()


def test_ingest_stores_child_chunks_with_embeddings(
    pipeline,
    vector_store,
):
    pipeline.ingest("python.txt")

    call_kwargs = vector_store.add_documents.call_args.kwargs

    assert call_kwargs["documents"] == [
        "Python is a programming language."
    ]

    assert call_kwargs["embeddings"] == [
        [0.1, 0.2, 0.3]
    ]

    assert len(call_kwargs["ids"]) == 1
    assert len(set(call_kwargs["ids"])) == 1


def test_ingest_stores_parent_chunks_with_embeddings(
    pipeline,
    vector_store,
):
    pipeline.ingest("python.txt")

    call_kwargs = (
        vector_store.add_parent_documents.call_args.kwargs
    )

    assert call_kwargs["documents"] == [
        "Python is a programming language."
    ]

    assert call_kwargs["embeddings"] == [
        [0.1, 0.2, 0.3]
    ]

    assert len(call_kwargs["ids"]) == 1
    assert len(set(call_kwargs["ids"])) == 1


def test_ingest_stores_child_metadata(
    pipeline,
    vector_store,
):
    pipeline.ingest("python.txt")

    call_kwargs = vector_store.add_documents.call_args.kwargs
    metadata = call_kwargs["metadatas"][0]

    assert metadata["source"] == "C:/documents/python.txt"
    assert metadata["file_name"] == "python.txt"
    assert metadata["file_type"] == "txt"
    assert metadata["parent_index"] == 0
    assert metadata["chunk_index"] == 0

    assert metadata["document_id"]
    assert metadata["parent_id"]


def test_ingest_stores_parent_metadata(
    pipeline,
    vector_store,
):
    pipeline.ingest("python.txt")

    call_kwargs = (
        vector_store.add_parent_documents.call_args.kwargs
    )
    metadata = call_kwargs["metadatas"][0]

    assert metadata["source"] == "C:/documents/python.txt"
    assert metadata["file_name"] == "python.txt"
    assert metadata["file_type"] == "txt"
    assert metadata["parent_index"] == 0
    assert metadata["document_id"]


def test_ingest_multiple_child_chunks(
    pipeline,
    chunker,
    embedder,
    vector_store,
):
    chunks = [
        "First chunk.",
        "Second chunk.",
        "Third chunk.",
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    chunker.chunk_text.return_value = chunks
    embedder.embed_texts.return_value = embeddings

    result = pipeline.ingest("python.txt")

    child_kwargs = vector_store.add_documents.call_args.kwargs
    parent_kwargs = (
        vector_store.add_parent_documents.call_args.kwargs
    )

    assert result["chunk_count"] == 3
    assert result["child_count"] == 3
    assert result["parent_count"] == 1

    assert len(result["child_ids"]) == 3
    assert len(set(result["child_ids"])) == 3

    assert child_kwargs["documents"] == chunks
    assert child_kwargs["embeddings"] == embeddings

    assert len(child_kwargs["metadatas"]) == 3
    assert [
        item["chunk_index"]
        for item in child_kwargs["metadatas"]
    ] == [0, 1, 2]

    assert len(parent_kwargs["documents"]) == 1
    assert len(parent_kwargs["ids"]) == 1

    assert child_kwargs["metadatas"][0]["parent_id"] == (
        parent_kwargs["ids"][0]
    )


def test_ingest_empty_document_does_not_store_data(
    pipeline,
    loader,
    chunker,
    embedder,
    vector_store,
):
    loader.load.return_value = {
        "content": "",
        "metadata": {
            "source": "C:/documents/empty.txt",
            "file_name": "empty.txt",
            "file_type": "txt",
        },
    }

    result = pipeline.ingest("empty.txt")

    assert result["chunk_count"] == 0
    assert result["parent_count"] == 0
    assert result["child_count"] == 0
    assert result["parent_ids"] == []
    assert result["child_ids"] == []

    chunker.chunk_text.assert_not_called()
    embedder.embed_texts.assert_not_called()
    vector_store.add_documents.assert_not_called()
    vector_store.add_parent_documents.assert_not_called()


def test_ingest_propagates_loader_error(
    pipeline,
    loader,
    chunker,
    embedder,
    vector_store,
):
    loader.load.side_effect = FileNotFoundError(
        "Document not found"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Document not found",
    ):
        pipeline.ingest("missing.txt")

    chunker.chunk_text.assert_not_called()
    embedder.embed_texts.assert_not_called()
    vector_store.add_documents.assert_not_called()
    vector_store.add_parent_documents.assert_not_called()


