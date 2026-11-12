import pytest

from cognexa.knowledge.retrieval.contracts.document import DocumentChunk
from cognexa.knowledge.retrieval.contracts.evidence import (
    Evidence,
    EvidenceType,
)
from cognexa.knowledge.retrieval.evidence_context_builder import (
    EvidenceContextBuilder,
)


@pytest.fixture
def builder():
    return EvidenceContextBuilder()


def make_evidence(
    chunk_id,
    content,
    evidence_type,
    position=None,
    metadata=None,
):
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        content=content,
        metadata=metadata or {},
        position=position,
    )

    return Evidence(
        evidence_id=f"evidence_{chunk_id}",
        chunk=chunk,
        evidence_type=evidence_type,
        source_chunk_id=chunk_id,
    )


def test_build_empty_evidence(builder):
    assert builder.build([]) == ""


def test_build_includes_content_and_source(builder):
    evidence = make_evidence(
        chunk_id="chunk_1",
        content="Python is a programming language.",
        evidence_type=EvidenceType.PRIMARY,
        metadata={"file_name": "python.txt"},
    )

    context = builder.build([evidence])

    assert "Python is a programming language." in context
    assert "Source: python.txt" in context
    assert "Type: primary" in context


def test_build_prioritizes_evidence_types(builder):
    neighbor = make_evidence(
        "chunk_3",
        "Neighbor content.",
        EvidenceType.NEIGHBOR,
    )
    parent = make_evidence(
        "chunk_2",
        "Parent content.",
        EvidenceType.PARENT,
    )
    primary = make_evidence(
        "chunk_1",
        "Primary content.",
        EvidenceType.PRIMARY,
    )

    context = builder.build([neighbor, parent, primary])

    assert context.index("Primary content.") < context.index("Parent content.")
    assert context.index("Parent content.") < context.index("Neighbor content.")


def test_build_deduplicates_by_chunk_id(builder):
    primary = make_evidence(
        "chunk_1",
        "Primary content.",
        EvidenceType.PRIMARY,
    )
    neighbor = make_evidence(
        "chunk_1",
        "Neighbor content.",
        EvidenceType.NEIGHBOR,
    )

    context = builder.build([neighbor, primary])

    assert context.count("Primary content.") == 1
    assert "Neighbor content." not in context
    assert context.count("[Document") == 1


def test_build_orders_same_type_by_position(builder):
    later = make_evidence(
        "chunk_2",
        "Second content.",
        EvidenceType.PRIMARY,
        position=2,
    )
    earlier = make_evidence(
        "chunk_1",
        "First content.",
        EvidenceType.PRIMARY,
        position=1,
    )

    context = builder.build([later, earlier])

    assert context.index("First content.") < context.index("Second content.")


def test_build_uses_source_fallback(builder):
    evidence = make_evidence(
        chunk_id="chunk_1",
        content="Some content.",
        evidence_type=EvidenceType.PRIMARY,
    )

    context = builder.build([evidence])

    assert "Source: doc_1" in context


def test_build_rejects_invalid_input(builder):
    with pytest.raises(TypeError):
        builder.build("invalid")

    with pytest.raises(TypeError):
        builder.build(["invalid"])


def test_build_rejects_invalid_metadata(builder):
    evidence = make_evidence(
        chunk_id="chunk_1",
        content="Some content.",
        evidence_type=EvidenceType.PRIMARY,
        metadata={"file_name": "   ", "source": "valid.txt"},
    )

    context = builder.build([evidence])

    # A blank file_name should not hide the valid source fallback.
    assert "Source: valid.txt" in context
