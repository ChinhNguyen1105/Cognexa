import pytest

from cognexa.knowledge.retrieval.candidates.sparse_retriever import (
    BM25Retriever,
    tokenize,
)
from cognexa.knowledge.retrieval.contracts import (
    DocumentChunk,
    RetrievalQuery,
    RetrieverType,
)


def make_documents():
    return [
        DocumentChunk(
            chunk_id="python-001",
            document_id="python-doc",
            content=(
                "Python is a high level programming language. "
                "Python is widely used for software development."
            ),
        ),
        DocumentChunk(
            chunk_id="python-002",
            document_id="python-doc",
            content=(
                "Python supports object oriented programming "
                "and dynamic typing."
            ),
        ),
        DocumentChunk(
            chunk_id="java-001",
            document_id="java-doc",
            content=(
                "Java is a programming language used for "
                "enterprise software development."
            ),
        ),
        DocumentChunk(
            chunk_id="database-001",
            document_id="database-doc",
            content=(
                "Relational databases organize data into "
                "tables, rows and columns."
            ),
        ),
    ]


def make_query(text="Python programming language"):
    return RetrievalQuery(
        query_id="q1",
        text=text,
    )


def test_tokenize_is_case_insensitive():
    assert tokenize("Python PROGRAMMING") == [
        "python",
        "programming",
    ]


def test_bm25_retriever_creation():
    retriever = BM25Retriever(make_documents())

    assert retriever.document_count == 4
    assert retriever.indexed_document_count == 4


def test_bm25_retriever_returns_retrieval_run():
    retriever = BM25Retriever(make_documents())

    result = retriever.retrieve(make_query(), top_k=3)

    assert result.query_id == "q1"
    assert result.retriever == RetrieverType.BM25
    assert len(result.hits) > 0


def test_bm25_retriever_ranks_relevant_document():
    retriever = BM25Retriever(make_documents())

    result = retriever.retrieve(make_query(), top_k=4)

    assert result.hits[0].chunk.chunk_id in {
        "python-001",
        "python-002",
    }


def test_bm25_retriever_preserves_score_information():
    retriever = BM25Retriever(make_documents())

    result = retriever.retrieve(make_query())

    hit = result.hits[0]

    assert hit.rank == 1
    assert hit.raw_score > 0
    assert hit.score_type == "bm25"


def test_bm25_retriever_returns_empty_for_unmatched_query():
    retriever = BM25Retriever(make_documents())

    result = retriever.retrieve(
        make_query("quantum mechanics astrophysics"),
    )

    assert result.hits == []


def test_bm25_retriever_returns_empty_for_empty_corpus():
    retriever = BM25Retriever()

    result = retriever.retrieve(make_query())

    assert result.hits == []
    assert retriever.document_count == 0


def test_bm25_retriever_rejects_invalid_query():
    retriever = BM25Retriever(make_documents())

    with pytest.raises(TypeError):
        retriever.retrieve("Python programming")


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_bm25_retriever_rejects_invalid_top_k(top_k):
    retriever = BM25Retriever(make_documents())

    with pytest.raises(ValueError):
        retriever.retrieve(make_query(), top_k=top_k)


def test_bm25_retriever_rejects_duplicate_chunk_ids():
    documents = make_documents()

    with pytest.raises(ValueError):
        BM25Retriever([documents[0], documents[0]])


def test_bm25_retriever_upserts_documents():
    retriever = BM25Retriever(make_documents())

    updated_document = DocumentChunk(
        chunk_id="python-001",
        document_id="python-doc",
        content="Python is a language for machine learning.",
    )

    new_document = DocumentChunk(
        chunk_id="ml-001",
        document_id="ml-doc",
        content="Machine learning uses statistical algorithms.",
    )

    retriever.upsert_documents([
        updated_document,
        new_document,
    ])

    assert retriever.document_count == 5

    result = retriever.retrieve(
        make_query("machine learning"),
        top_k=5,
    )

    assert result.hits[0].chunk.chunk_id in {
        "python-001",
        "ml-001",
    }


def test_bm25_retriever_removes_documents():
    retriever = BM25Retriever(make_documents())

    removed = retriever.remove_documents(["python-001"])

    assert removed == 1
    assert retriever.document_count == 3

    result = retriever.retrieve(
        make_query("Python"),
        top_k=5,
    )

    assert all(
        hit.chunk.chunk_id != "python-001"
        for hit in result.hits
    )


def test_bm25_retriever_rejects_invalid_document_type():
    with pytest.raises(TypeError):
        BM25Retriever(["not-a-document"])


def test_bm25_retriever_rebuilds_after_replacing_documents():
    retriever = BM25Retriever(make_documents())

    replacement = DocumentChunk(
        chunk_id="rust-001",
        document_id="rust-doc",
        content="Rust is a systems programming language.",
    )

    retriever.replace_documents([replacement])

    assert retriever.document_count == 1

    result = retriever.retrieve(
        make_query("Python"),
        top_k=5,
    )

    assert result.hits == []