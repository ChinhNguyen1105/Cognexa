import pytest

from cognexa.knowledge.vector_store.store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        persist_directory=tmp_path / "vector_db",
        collection_name="test_collection",
    )


def test_create_vector_store(store):
    assert store.count() == 0


def test_add_documents(store):
    store.add_documents(
        ids=["doc_1", "doc_2"],
        documents=[
            "Python is a programming language.",
            "NestJS is a Node.js framework.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    assert store.count() == 2


def test_add_documents_with_metadata(store):
    store.add_documents(
        ids=["doc_1"],
        documents=["Python is a programming language."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[
            {
                "source": "python.txt",
                "file_type": "txt",
            }
        ],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert results[0]["metadata"]["source"] == "python.txt"
    assert results[0]["metadata"]["file_type"] == "txt"


def test_search_returns_most_similar_document(store):
    store.add_documents(
        ids=["doc_1", "doc_2"],
        documents=[
            "Python programming language.",
            "Database management system.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc_1"
    assert results[0]["content"] == "Python programming language."
    assert results[0]["distance"] == pytest.approx(0.0)


def test_search_returns_empty_list_when_store_is_empty(store):
    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert results == []


def test_search_respects_top_k(store):
    store.add_documents(
        ids=["doc_1", "doc_2", "doc_3"],
        documents=[
            "Document one.",
            "Document two.",
            "Document three.",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_delete_documents(store):
    store.add_documents(
        ids=["doc_1", "doc_2"],
        documents=["Document one.", "Document two."],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    store.delete(["doc_1"])

    assert store.count() == 1

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert all(result["id"] != "doc_1" for result in results)


def test_data_persists_after_reopening_store(tmp_path):
    persist_directory = tmp_path / "vector_db"

    first_store = VectorStore(
        persist_directory=persist_directory,
        collection_name="persistent_collection",
    )

    first_store.add_documents(
        ids=["doc_1"],
        documents=["Cognexa knowledge document."],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    second_store = VectorStore(
        persist_directory=persist_directory,
        collection_name="persistent_collection",
    )

    assert second_store.count() == 1


def test_reset_clears_collection(store):
    store.add_documents(
        ids=["doc_1"],
        documents=["Document one."],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    store.reset()

    assert store.count() == 0


def test_add_documents_rejects_empty_ids(store):
    with pytest.raises(ValueError, match="ids cannot be empty"):
        store.add_documents(
            ids=[],
            documents=[],
            embeddings=[],
        )


def test_add_documents_rejects_different_lengths(store):
    with pytest.raises(ValueError, match="equal lengths"):
        store.add_documents(
            ids=["doc_1", "doc_2"],
            documents=["Document one."],
            embeddings=[[1.0, 0.0, 0.0]],
        )


def test_add_documents_rejects_duplicate_ids(store):
    with pytest.raises(ValueError, match="unique"):
        store.add_documents(
            ids=["doc_1", "doc_1"],
            documents=["Document one.", "Document two."],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )


def test_add_documents_rejects_different_embedding_dimensions(store):
    with pytest.raises(ValueError, match="same dimensions"):
        store.add_documents(
            ids=["doc_1", "doc_2"],
            documents=["Document one.", "Document two."],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0],
            ],
        )


def test_add_documents_rejects_metadata_length_mismatch(store):
    with pytest.raises(ValueError, match="same length"):
        store.add_documents(
            ids=["doc_1", "doc_2"],
            documents=["Document one.", "Document two."],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            metadatas=[{"source": "document.txt"}],
        )


def test_add_documents_rejects_invalid_metadata(store):
    with pytest.raises(TypeError, match="dictionary"):
        store.add_documents(
            ids=["doc_1"],
            documents=["Document one."],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=["invalid metadata"],
        )


def test_search_rejects_empty_query_embedding(store):
    with pytest.raises(ValueError, match="cannot be empty"):
        store.search(
            query_embedding=[],
        )


def test_search_rejects_invalid_top_k(store):
    with pytest.raises(ValueError, match="greater than 0"):
        store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=0,
        )


def test_delete_rejects_empty_ids(store):
    with pytest.raises(ValueError, match="ids cannot be empty"):
        store.delete([])
