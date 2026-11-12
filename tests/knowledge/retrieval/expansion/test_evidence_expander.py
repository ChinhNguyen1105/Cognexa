import pytest

from cognexa.knowledge.retrieval.contracts.candidate import (
    FusedCandidate,
    RankedCandidate,
    RetrievalMatch,
    RetrieverType,
)
from cognexa.knowledge.retrieval.contracts.document import DocumentChunk
from cognexa.knowledge.retrieval.contracts.evidence import EvidenceType
from cognexa.knowledge.retrieval.expansion.evidence_expander import (
    EvidenceExpander,
)


class FakeVectorStore:
    def __init__(
        self,
        parents: dict[str, dict] | None = None,
        children: dict[str, list[dict]] | None = None,
    ):
        self.parents = parents or {}
        self.children = children or {}
        self.parent_calls = []
        self.child_calls = []

    def get_parent_documents(self, ids: list[str]) -> list[dict]:
        self.parent_calls.append(ids)

        return [
            self.parents[parent_id]
            for parent_id in ids
            if parent_id in self.parents
        ]

    def get_child_documents(self, parent_id: str) -> list[dict]:
        self.child_calls.append(parent_id)
        return self.children.get(parent_id, [])


def make_chunk(
    chunk_id: str,
    parent_id: str | None = "parent_0",
    position: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document_1",
        content=f"Content of {chunk_id}",
        metadata={
            "document_id": "document_1",
            "parent_id": parent_id,
            "chunk_index": position,
        },
        parent_chunk_id=parent_id,
        position=position,
    )


def make_store_chunk(
    chunk_id: str,
    parent_id: str,
    position: int,
) -> dict:
    return {
        "id": chunk_id,
        "content": f"Content of {chunk_id}",
        "metadata": {
            "document_id": "document_1",
            "parent_id": parent_id,
            "chunk_index": position,
        },
    }


def make_ranked_candidate(
    chunk: DocumentChunk,
    query_id: str = "query_1",
    score: float = 0.9,
    rank: int = 1,
) -> RankedCandidate:
    fused = FusedCandidate(
        chunk=chunk,
        rrf_score=0.05,
        matches=[
            RetrievalMatch(
                query_id=query_id,
                retriever=RetrieverType.DENSE,
                rank=rank,
                raw_score=0.05,
                score_type="distance",
            )
        ],
        fusion_rank=rank,
    )

    return RankedCandidate(
        query_id=query_id,
        candidate=fused,
        rerank_score=score,
        rerank_rank=rank,
    )


def make_parent(parent_id: str = "parent_0") -> dict:
    return {
        "id": parent_id,
        "content": f"Content of {parent_id}",
        "metadata": {
            "document_id": "document_1",
            "parent_index": 0,
        },
    }


def test_expand_returns_primary_parent_and_neighbors():
    source = make_chunk("child_1", position=1)

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={
            "parent_0": [
                make_store_chunk("child_0", "parent_0", 0),
                make_store_chunk("child_1", "parent_0", 1),
                make_store_chunk("child_2", "parent_0", 2),
            ]
        },
    )

    expander = EvidenceExpander(store)

    evidence = expander.expand([
        make_ranked_candidate(source),
    ])

    assert [item.chunk.chunk_id for item in evidence] == [
        "child_1",
        "parent_0",
        "child_0",
        "child_2",
    ]

    assert [item.evidence_type for item in evidence] == [
        EvidenceType.PRIMARY,
        EvidenceType.PARENT,
        EvidenceType.NEIGHBOR,
        EvidenceType.NEIGHBOR,
    ]

    assert evidence[0].relevance_score == 0.9
    assert evidence[1].relevance_score == 0.9
    assert evidence[2].relevance_score is None
    assert evidence[3].relevance_score is None


def test_expand_does_not_cross_parent_boundary():
    source = make_chunk("child_1", position=1)

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={
            "parent_0": [
                make_store_chunk("child_0", "parent_0", 0),
                make_store_chunk("child_1", "parent_0", 1),
            ],
            "parent_1": [
                make_store_chunk("child_2", "parent_1", 0),
            ],
        },
    )

    evidence = EvidenceExpander(store).expand([
        make_ranked_candidate(source),
    ])

    evidence_ids = {item.chunk.chunk_id for item in evidence}

    assert "child_2" not in evidence_ids
    assert store.child_calls == ["parent_0"]


def test_primary_has_priority_over_neighbor():
    chunk_a = make_chunk("child_a", position=0)
    chunk_b = make_chunk("child_b", position=1)

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={
            "parent_0": [
                make_store_chunk("child_a", "parent_0", 0),
                make_store_chunk("child_b", "parent_0", 1),
            ]
        },
    )

    evidence = EvidenceExpander(store).expand([
        make_ranked_candidate(
            chunk_a,
            query_id="query_1",
            score=0.8,
            rank=1,
        ),
        make_ranked_candidate(
            chunk_b,
            query_id="query_2",
            score=0.95,
            rank=2,
        ),
    ])

    evidence_by_id = {
        item.chunk.chunk_id: item
        for item in evidence
    }

    assert len(evidence_by_id) == len(evidence)

    assert evidence_by_id["child_a"].evidence_type == EvidenceType.PRIMARY
    assert evidence_by_id["child_b"].evidence_type == EvidenceType.PRIMARY


def test_duplicate_evidence_merges_query_ids_and_keeps_highest_score():
    source = make_chunk("child_1", position=1)

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={
            "parent_0": [
                make_store_chunk("child_0", "parent_0", 0),
                make_store_chunk("child_1", "parent_0", 1),
                make_store_chunk("child_2", "parent_0", 2),
            ]
        },
    )

    evidence = EvidenceExpander(store).expand([
        make_ranked_candidate(
            source,
            query_id="query_1",
            score=0.8,
            rank=1,
        ),
        make_ranked_candidate(
            source,
            query_id="query_2",
            score=0.95,
            rank=2,
        ),
    ])

    primary = next(
        item for item in evidence
        if item.chunk.chunk_id == "child_1"
    )

    parent = next(
        item for item in evidence
        if item.chunk.chunk_id == "parent_0"
    )

    assert primary.related_query_ids == ["query_1", "query_2"]
    assert primary.relevance_score == 0.95

    assert parent.related_query_ids == ["query_1", "query_2"]
    assert parent.relevance_score == 0.95

    assert len(evidence) == 4


def test_expand_keeps_primary_when_parent_is_missing():
    source = make_chunk("child_1")

    store = FakeVectorStore(
        parents={},
        children={
            "parent_0": [
                make_store_chunk("child_1", "parent_0", 0),
            ]
        },
    )

    evidence = EvidenceExpander(store).expand([
        make_ranked_candidate(source),
    ])

    assert len(evidence) == 1
    assert evidence[0].chunk.chunk_id == "child_1"
    assert evidence[0].evidence_type == EvidenceType.PRIMARY


def test_neighbor_window_zero_disables_neighbor_retrieval():
    source = make_chunk("child_1")

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={
            "parent_0": [
                make_store_chunk("child_1", "parent_0", 0),
            ]
        },
    )

    evidence = EvidenceExpander(
        store,
        neighbor_window=0,
    ).expand([
        make_ranked_candidate(source),
    ])

    assert [item.evidence_type for item in evidence] == [
        EvidenceType.PRIMARY,
        EvidenceType.PARENT,
    ]

    assert store.child_calls == []


def test_neighbor_window_cannot_be_negative():
    store = FakeVectorStore()

    with pytest.raises(ValueError):
        EvidenceExpander(
            store,
            neighbor_window=-1,
        )


def test_empty_candidates_return_empty_evidence():
    store = FakeVectorStore()

    evidence = EvidenceExpander(store).expand([])

    assert evidence == []
    assert store.parent_calls == []
    assert store.child_calls == []



def test_expand_retrieves_distant_list_siblings():
    source = make_chunk("child_1", position=1)
    source.metadata["list_groups"] = [
        {"group_id": 0, "item_indices": [1]}
    ]

    children = [
        make_store_chunk("child_0", "parent_0", 0),
        make_store_chunk("child_1", "parent_0", 1),
        make_store_chunk("child_2", "parent_0", 2),
        make_store_chunk("child_3", "parent_0", 3),
    ]

    children[0]["metadata"]["list_groups"] = [
        {"group_id": 0, "item_indices": [0]}
    ]
    children[1]["metadata"]["list_groups"] = [
        {"group_id": 0, "item_indices": [1]}
    ]
    children[2]["metadata"]["list_groups"] = [
        {"group_id": 0, "item_indices": [2]}
    ]

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={"parent_0": children},
    )

    evidence = EvidenceExpander(
        store,
        neighbor_window=0,
    ).expand([make_ranked_candidate(source)])

    neighbor_ids = [
        item.chunk.chunk_id
        for item in evidence
        if item.evidence_type == EvidenceType.NEIGHBOR
    ]

    assert neighbor_ids == ["child_0", "child_2"]
    assert "child_3" not in neighbor_ids


def test_expand_does_not_include_unrelated_list_groups():
    source = make_chunk("child_1", position=1)
    source.metadata["list_groups"] = [
        {"group_id": 0, "item_indices": [1]}
    ]

    children = [
        make_store_chunk("child_1", "parent_0", 1),
        make_store_chunk("child_2", "parent_0", 2),
    ]

    children[1]["metadata"]["list_groups"] = [
        {"group_id": 1, "item_indices": [2]}
    ]

    store = FakeVectorStore(
        parents={"parent_0": make_parent()},
        children={"parent_0": children},
    )

    evidence = EvidenceExpander(
        store,
        neighbor_window=0,
    ).expand([make_ranked_candidate(source)])

    neighbor_ids = [
        item.chunk.chunk_id
        for item in evidence
        if item.evidence_type == EvidenceType.NEIGHBOR
    ]

    assert neighbor_ids == []
