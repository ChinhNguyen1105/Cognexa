import pytest

from cognexa.knowledge.retrieval.contracts import DocumentChunk


def test_document_chunk_creation():
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="python-doc",
        content="Python is a programming language.",
        metadata={"topic": "python"},
        position=0,
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.document_id == "python-doc"
    assert chunk.metadata["topic"] == "python"
    assert chunk.position == 0


def test_document_chunk_supports_parent():
    chunk = DocumentChunk(
        chunk_id="child-001",
        document_id="python-doc",
        content="Python supports dynamic typing.",
        parent_chunk_id="parent-001",
    )

    assert chunk.parent_chunk_id == "parent-001"


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", ""),
        ("document_id", ""),
        ("content", ""),
    ],
)
def test_document_chunk_rejects_empty_required_fields(field, value):
    data = {
        "chunk_id": "chunk-001",
        "document_id": "doc-001",
        "content": "Some content",
    }

    data[field] = value

    with pytest.raises(ValueError):
        DocumentChunk(**data)


def test_document_chunk_rejects_invalid_metadata():
    with pytest.raises(TypeError):
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Some content",
            metadata=[],
        )


def test_document_chunk_rejects_negative_position():
    with pytest.raises(ValueError):
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Some content",
            position=-1,
        )