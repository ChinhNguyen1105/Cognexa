from __future__ import annotations

import pytest

from cognexa.knowledge.retrieval.adaptive_retriever import AdaptiveRetriever
from cognexa.knowledge.retrieval.candidates.dense_retriever import (
    DenseRetriever,
)
from cognexa.knowledge.retrieval.candidates.sparse_retriever import (
    BM25Retriever,
)
from cognexa.knowledge.retrieval.fusion.rrf_fusion import (
    ReciprocalRankFusion,
)
from cognexa.knowledge.retrieval.ranking.cross_encoder_reranker import (
    CrossEncoderCandidateReranker,
)
from cognexa.knowledge.retrieval.ranking.reranker import (
    CrossEncoderReranker,
)
from cognexa.knowledge.retrieval.expansion.evidence_expander import (
    EvidenceExpander,
)
from cognexa.knowledge.retrieval.coverage.coverage_analyzer import (
    CoverageAnalyzer,
)
from cognexa.knowledge.retrieval.coverage.sufficiency_checker import (
    SufficiencyChecker,
)
from cognexa.knowledge.retrieval.contracts.document import DocumentChunk
from cognexa.knowledge.retrieval.contracts.query import (
    QueryPlan,
    RetrievalQuery,
)
from cognexa.knowledge.retrieval.contracts.evidence import EvidenceType
from cognexa.knowledge.retrieval.contracts.result import (
    RetrievalStopReason,
)
from cognexa.knowledge.vector_store.store import VectorStore


class FakeEmbedder:
    """Deterministic embedder for integration testing."""

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeCrossEncoder(CrossEncoderReranker):
    """Replace the real Cross-Encoder model with deterministic scores."""

    def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[dict]:
        results = [
            {
                "index": index,
                "score": 1.0 - index * 0.1,
            }
            for index in range(len(candidates))
        ]

        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )


@pytest.fixture
def vector_store(tmp_path):
    return VectorStore(
        persist_directory=tmp_path / "adaptive_retrieval_db",
        collection_name="adaptive_retrieval_integration",
    )


@pytest.fixture
def chunks():
    parent_id = "parent_python"

    return [
        DocumentChunk(
            chunk_id="python_0",
            document_id="python_doc",
            content=(
                "Python is a high-level programming language. "
                "It is widely used for software development."
            ),
            metadata={
                "document_id": "python_doc",
                "parent_id": parent_id,
                "chunk_index": 0,
            },
            parent_chunk_id=parent_id,
            position=0,
        ),
        DocumentChunk(
            chunk_id="python_1",
            document_id="python_doc",
            content=(
                "Python supports object-oriented programming "
                "and functional programming."
            ),
            metadata={
                "document_id": "python_doc",
                "parent_id": parent_id,
                "chunk_index": 1,
            },
            parent_chunk_id=parent_id,
            position=1,
        ),
        DocumentChunk(
            chunk_id="python_2",
            document_id="python_doc",
            content=(
                "Python provides libraries for data science "
                "and machine learning."
            ),
            metadata={
                "document_id": "python_doc",
                "parent_id": parent_id,
                "chunk_index": 2,
            },
            parent_chunk_id=parent_id,
            position=2,
        ),
    ]


def populate_vector_store(store, chunks):
    parent_id = "parent_python"

    store.add_parent_documents(
        ids=[parent_id],
        documents=[
            (
                "Python is a general-purpose programming language "
                "used in software development, data science, "
                "and machine learning."
            )
        ],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[
            {
                "document_id": "python_doc",
                "parent_index": 0,
            }
        ],
    )

    store.add_documents(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        embeddings=[
            [1.0, 0.0, 0.0]
            for _ in chunks
        ],
        metadatas=[
            chunk.metadata
            for chunk in chunks
        ],
    )


def test_adaptive_retriever_runs_complete_pipeline(
    vector_store,
    chunks,
):
    # Arrange: populate a real ChromaDB VectorStore.
    populate_vector_store(vector_store, chunks)

    # M1: real Dense and BM25 retrievers.
    dense_retriever = DenseRetriever(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    sparse_retriever = BM25Retriever(
        documents=chunks,
    )

    # M2: real RRF.
    fusion = ReciprocalRankFusion(rrf_k=60)

    # M3: real adapter, fake Cross-Encoder scorer.
    reranker = CrossEncoderCandidateReranker(
        scorer=FakeCrossEncoder(),
    )

    # M4: real EvidenceExpander and VectorStore.
    expander = EvidenceExpander(
        vector_store=vector_store,
        neighbor_window=1,
    )

    # M5: real coverage and sufficiency components.
    coverage_analyzer = CoverageAnalyzer()

    sufficiency_checker = SufficiencyChecker(
        min_coverage=1.0,
    )

    # M6: orchestrate all components.
    adaptive_retriever = AdaptiveRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
        reranker=reranker,
        expander=expander,
        coverage_analyzer=coverage_analyzer,
        sufficiency_checker=sufficiency_checker,
        top_k=1,
        max_iterations=3,
    )

    query_plan = QueryPlan(
        original_query="What is Python?",
        normalized_query="Python programming language",
        intent="information",
        retrieval_queries=[
            RetrievalQuery(
                query_id="q1",
                text="Python programming language",
            )
        ],
    )

    # Act.
    result = adaptive_retriever.retrieve(query_plan)

    # Assert: adaptive retrieval result.
    assert result.sufficient is True
    assert result.stop_reason == RetrievalStopReason.THRESHOLD_REACHED
    assert result.iterations == 1
    assert result.coverage.overall_score == 1.0

    # Assert: primary evidence exists.
    primary_evidence = [
        item
        for item in result.evidence
        if item.evidence_type == EvidenceType.PRIMARY
    ]

    assert len(primary_evidence) > 0

    # Assert: parent evidence was retrieved from ChromaDB.
    parent_evidence = [
        item
        for item in result.evidence
        if item.evidence_type == EvidenceType.PARENT
    ]

    assert len(parent_evidence) == 1
    assert parent_evidence[0].chunk.chunk_id == "parent_python"

    # Assert: neighboring evidence was expanded.
    neighbor_evidence = [
        item
        for item in result.evidence
        if item.evidence_type == EvidenceType.NEIGHBOR
    ]

    assert len(neighbor_evidence) > 0

    # Assert: coverage traceability is preserved.
    assert all(
        "q1" in item.related_query_ids
        for item in result.evidence
    )

    # Assert: evidence IDs are unique.
    evidence_ids = [
        item.evidence_id
        for item in result.evidence
    ]

    assert len(evidence_ids) == len(set(evidence_ids))

    # Assert: diagnostics are populated.
    history = result.diagnostics["iteration_history"]

    assert len(history) == 1
    assert history[0]["top_k"] == 1
    assert history[0]["sufficient"] is True


