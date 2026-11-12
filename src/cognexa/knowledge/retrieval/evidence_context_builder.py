from collections.abc import Sequence

from cognexa.knowledge.retrieval.contracts.evidence import (
    Evidence,
    EvidenceType,
)


class EvidenceContextBuilder:
    """
    Convert retrieval evidence into a context string
    compatible with AnswerExtractor.
    """

    _TYPE_PRIORITY = {
        EvidenceType.PRIMARY: 3,
        EvidenceType.PARENT: 2,
        EvidenceType.NEIGHBOR: 1,
    }

    def build(self, evidence: Sequence[Evidence]) -> str:
        if not isinstance(evidence, (list, tuple)):
            raise TypeError("evidence must be a list or tuple")

        unique_evidence = {}

        for item in evidence:
            if not isinstance(item, Evidence):
                raise TypeError("Every item must be an Evidence instance")

            chunk_id = item.chunk.chunk_id
            existing = unique_evidence.get(chunk_id)

            if (
                existing is None
                or self._TYPE_PRIORITY[item.evidence_type]
                > self._TYPE_PRIORITY[existing.evidence_type]
            ):
                unique_evidence[chunk_id] = item

        ordered_evidence = sorted(
            unique_evidence.values(),
            key=lambda item: (
                -self._TYPE_PRIORITY[item.evidence_type],
                item.chunk.position
                if item.chunk.position is not None
                else float("inf"),
                item.chunk.chunk_id,
            ),
        )

        sections = []

        for index, item in enumerate(ordered_evidence, start=1):
            chunk = item.chunk
            metadata = chunk.metadata

            source = None

            for candidate in (
                metadata.get("file_name"),
                metadata.get("source"),
            ):
                if isinstance(candidate, str) and candidate.strip():
                    source = candidate.strip()
                    break

            if source is None:
                source = chunk.document_id

            sections.append(
                f"[Document {index} | Source: {source} "
                f"| Type: {item.evidence_type.value}]\n"
                f"{chunk.content.strip()}"
            )

        return "\n\n".join(sections)
