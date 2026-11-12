import pytest

from cognexa.knowledge.retrieval.contracts.candidate import (
    FusedCandidate,
    RankedCandidate,
    RetrievalMatch,
    RetrieverType,
)
from cognexa.knowledge.retrieval.contracts.document import DocumentChunk
from cognexa.knowledge.retrieval.contracts.evidence import EvidenceType
from cognexa.knowledge.retrieval.expansion.evidence_expander import EvidenceExpander
from cognexa.knowledge.vector_store.store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        persist_directory=tmp_path / "vector_db",
        collection_name="expansion_integration_test",
    )


def make_ranked_candidate(
    chunk_id: str,
    parent_id: str,
    content: str,
    chunk_index: int,
    query_id: str = "query_1",
    score: float = 0.9,
) -> RankedCandidate:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document_1",
        content=content,
        metadata={
            "document_id": "document_1",
            "parent_id": parent_id,
            "chunk_index": chunk_index,
        },
        parent_chunk_id=parent_id,
        position=chunk_index,
    )

    fused = FusedCandidate(
        chunk=chunk,
        rrf_score=0.05,
        matches=[
            RetrievalMatch(
                query_id=query_id,
                retriever=RetrieverType.DENSE,
                rank=1,
                raw_score=0.05,
                score_type="distance",
            )
        ],
        fusion_rank=1,
    )

    return RankedCandidate(
        query_id=query_id,
        candidate=fused,
        rerank_score=score,
        rerank_rank=1,
    )


def add_parent(store, parent_id: str, content: str) -> None:
    store.add_parent_documents(
        ids=[parent_id],
        documents=[content],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[
            {
                "document_id": "document_1",
                "parent_index": 0,
            }
        ],
    )


def add_children(store, children: list[dict]) -> None:
    store.add_documents(
        ids=[item["id"] for item in children],
        documents=[item["content"] for item in children],
        embeddings=[
            [1.0, 0.0, 0.0] for _ in children
        ],
        metadatas=[
            {
                "document_id": item["document_id"],
                "parent_id": item["parent_id"],
                "chunk_index": item["chunk_index"],
            }
            for item in children
        ],
    )


def test_expand_retrieves_parent_from_real_vector_store(store):
    parent_id = "parent_1"

    add_parent(
        store,
        parent_id,
        "This is the complete parent document.",
    )

    add_children(
        store,
        [
            {
                "id": "child_1",
                "content": "This is child one.",
                "document_id": "document_1",
                "parent_id": parent_id,
                "chunk_index": 0,
            }
        ],
    )

    candidate = make_ranked_candidate(
        chunk_id="child_1",
        parent_id=parent_id,
        content="This is child one.",
        chunk_index=0,
    )

    expander = EvidenceExpander(store, neighbor_window=0)

    evidence = expander.expand([candidate])

    parent_evidence = next(
        item for item in evidence
        if item.evidence_type == EvidenceType.PARENT
    )

    assert parent_evidence.chunk.chunk_id == parent_id
    assert parent_evidence.chunk.content == (
        "This is the complete parent document."
    )


def test_expand_retrieves_neighbors_in_chunk_order(store):
    parent_id = "parent_1"

    add_parent(store, parent_id, "Parent document.")

    children = [
        {
            "id": "child_0",
            "content": "First chunk.",
            "document_id": "document_1",
            "parent_id": parent_id,
            "chunk_index": 0,
        },
        {
            "id": "child_1",
            "content": "Second chunk.",
            "document_id": "document_1",
            "parent_id": parent_id,
            "chunk_index": 1,
        },
        {
            "id": "child_2",
            "content": "Third chunk.",
            "document_id": "document_1",
            "parent_id": parent_id,
            "chunk_index": 2,
        },
    ]

    # Insert in a deliberately unordered sequence.
    add_children(store, [children[2], children[0], children[1]])

    candidate = make_ranked_candidate(
        chunk_id="child_1",
        parent_id=parent_id,
        content="Second chunk.",
        chunk_index=1,
    )

    expander = EvidenceExpander(store, neighbor_window=1)

    evidence = expander.expand([candidate])

    neighbors = [
        item.chunk.chunk_id
        for item in evidence
        if item.evidence_type == EvidenceType.NEIGHBOR
    ]

    assert neighbors == ["child_0", "child_2"]


def test_expand_does_not_retrieve_neighbors_from_another_parent(store):
    parent_a = "parent_a"
    parent_b = "parent_b"

    add_parent(store, parent_a, "Parent A.")
    add_parent(store, parent_b, "Parent B.")

    add_children(
        store,
        [
            {
                "id": "a_child_0",
                "content": "Parent A first chunk.",
                "document_id": "document_1",
                "parent_id": parent_a,
                "chunk_index": 0,
            },
            {
                "id": "a_child_1",
                "content": "Parent A second chunk.",
                "document_id": "document_1",
                "parent_id": parent_a,
                "chunk_index": 1,
            },
            {
                "id": "b_child_0",
                "content": "Parent B first chunk.",
                "document_id": "document_1",
                "parent_id": parent_b,
                "chunk_index": 0,
            },
        ],
    )

    candidate = make_ranked_candidate(
        chunk_id="a_child_1",
        parent_id=parent_a,
        content="Parent A second chunk.",
        chunk_index=1,
    )

    expander = EvidenceExpander(store, neighbor_window=2)

    evidence = expander.expand([candidate])

    evidence_ids = {
        item.chunk.chunk_id for item in evidence
    }

    assert "a_child_0" in evidence_ids
    assert "b_child_0" not in evidence_ids


def test_expand_keeps_primary_when_parent_does_not_exist(store):
    add_children(
        store,
        [
            {
                "id": "orphan_child",
                "content": "Child without a stored parent.",
                "document_id": "document_1",
                "parent_id": "missing_parent",
                "chunk_index": 0,
            }
        ],
    )

    candidate = make_ranked_candidate(
        chunk_id="orphan_child",
        parent_id="missing_parent",
        content="Child without a stored parent.",
        chunk_index=0,
    )

    expander = EvidenceExpander(store, neighbor_window=1)

    evidence = expander.expand([candidate])

    assert len(evidence) == 1
    assert evidence[0].evidence_type == EvidenceType.PRIMARY
    assert evidence[0].chunk.chunk_id == "orphan_child"
