import json
from collections.abc import Sequence
from typing import Any

from cognexa.knowledge.retrieval.contracts.candidate import RankedCandidate
from cognexa.knowledge.retrieval.contracts.document import DocumentChunk
from cognexa.knowledge.retrieval.contracts.evidence import Evidence, EvidenceType
from cognexa.knowledge.vector_store.store import VectorStore


class EvidenceExpander:
    """Expand ranked candidates with parent and neighboring chunks."""

    _TYPE_PRIORITY = {
        EvidenceType.NEIGHBOR: 1,
        EvidenceType.PARENT: 2,
        EvidenceType.PRIMARY: 3,
    }

    def __init__(
        self,
        vector_store: VectorStore,
        neighbor_window: int = 1,
    ) -> None:
        if neighbor_window < 0:
            raise ValueError("neighbor_window cannot be negative.")

        self.vector_store = vector_store
        self.neighbor_window = neighbor_window

    def expand(
        self,
        candidates: Sequence[RankedCandidate],
    ) -> list[Evidence]:
        """Create primary, parent, and neighboring evidence."""

        evidence_by_chunk_id: dict[str, Evidence] = {}

        parent_cache: dict[str, DocumentChunk | None] = {}
        children_cache: dict[str, list[DocumentChunk]] = {}

        for ranked in candidates:
            source_chunk = ranked.candidate.chunk

            primary = Evidence(
                evidence_id=source_chunk.chunk_id,
                chunk=source_chunk,
                evidence_type=EvidenceType.PRIMARY,
                source_chunk_id=source_chunk.chunk_id,
                related_query_ids=[ranked.query_id],
                relevance_score=ranked.rerank_score,
            )
            self._merge_evidence(evidence_by_chunk_id, primary)

            parent_id = source_chunk.parent_chunk_id

            if not parent_id:
                continue

            if parent_id not in parent_cache:
                parent_cache[parent_id] = self._get_parent(parent_id)

            parent_chunk = parent_cache[parent_id]

            if parent_chunk is not None:
                parent_evidence = Evidence(
                    evidence_id=parent_chunk.chunk_id,
                    chunk=parent_chunk,
                    evidence_type=EvidenceType.PARENT,
                    source_chunk_id=source_chunk.chunk_id,
                    related_query_ids=[ranked.query_id],
                    relevance_score=ranked.rerank_score,
                )
                self._merge_evidence(
                    evidence_by_chunk_id,
                    parent_evidence,
                )

            if (
                self.neighbor_window == 0
                and not self._parse_list_groups(
                    source_chunk.metadata.get("list_groups")
                )
            ):
                continue

            if parent_id not in children_cache:
                children_cache[parent_id] = [
                    self._to_document_chunk(item)
                    for item in self.vector_store.get_child_documents(
                        parent_id=parent_id,
                    )
                ]

            children = children_cache[parent_id]

            expanded_chunks = (
                self._get_neighbors(
                    source_chunk=source_chunk,
                    children=children,
                )
                if self.neighbor_window > 0
                else []
            )

            list_siblings = self._get_list_siblings(
                source_chunk=source_chunk,
                children=children,
            )

            expanded_ids = {
                chunk.chunk_id for chunk in expanded_chunks
            }

            for chunk in list_siblings:
                if chunk.chunk_id not in expanded_ids:
                    expanded_chunks.append(chunk)
                    expanded_ids.add(chunk.chunk_id)

            for neighbor_chunk in expanded_chunks:
                neighbor_evidence = Evidence(
                    evidence_id=neighbor_chunk.chunk_id,
                    chunk=neighbor_chunk,
                    evidence_type=EvidenceType.NEIGHBOR,
                    source_chunk_id=source_chunk.chunk_id,
                    related_query_ids=[ranked.query_id],
                    relevance_score=None,
                )
                self._merge_evidence(
                    evidence_by_chunk_id,
                    neighbor_evidence,
                )

        return list(evidence_by_chunk_id.values())

    def _get_parent(self, parent_id: str) -> DocumentChunk | None:
        """Retrieve and convert a parent chunk."""

        results = self.vector_store.get_parent_documents(
            ids=[parent_id],
        )

        if not results:
            return None

        return self._to_document_chunk(results[0])

    def _get_neighbors(
        self,
        source_chunk: DocumentChunk,
        children: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Get adjacent chunks within the same parent."""

        source_index = next(
            (
                index
                for index, chunk in enumerate(children)
                if chunk.chunk_id == source_chunk.chunk_id
            ),
            None,
        )

        if source_index is None:
            return []

        start = max(0, source_index - self.neighbor_window)
        end = min(
            len(children),
            source_index + self.neighbor_window + 1,
        )

        return [
            chunk
            for chunk in children[start:end]
            if chunk.chunk_id != source_chunk.chunk_id
        ]

    def _get_list_siblings(
        self,
        source_chunk: DocumentChunk,
        children: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Get chunks containing other items from the same list group."""

        source_groups = self._parse_list_groups(
            source_chunk.metadata.get("list_groups")
        )

        if not source_groups:
            return []

        source_group_items = {
            group_id: set(item_indices)
            for group_id, item_indices in source_groups
        }

        siblings = []

        for chunk in children:
            if chunk.chunk_id == source_chunk.chunk_id:
                continue

            child_groups = self._parse_list_groups(
                chunk.metadata.get("list_groups")
            )

            is_sibling = any(
                group_id in source_group_items
                for group_id, _ in child_groups
            )

            if is_sibling:
                siblings.append(chunk)

        return sorted(
            siblings,
            key=lambda chunk: (
                chunk.position
                if chunk.position is not None
                else float("inf")
            ),
        )

    @staticmethod
    def _parse_list_groups(value: Any) -> list[tuple[int, list[int]]]:
        """Parse list-group metadata stored as JSON or native data."""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return []

        if not isinstance(value, list):
            return []

        groups = []

        for group in value:
            if not isinstance(group, dict):
                continue

            group_id = group.get("group_id")
            item_indices = group.get("item_indices")

            if (
                isinstance(group_id, bool)
                or not isinstance(group_id, int)
                or not isinstance(item_indices, list)
            ):
                continue

            valid_indices = [
                index
                for index in item_indices
                if isinstance(index, int) and not isinstance(index, bool)
            ]

            if valid_indices:
                groups.append((group_id, valid_indices))

        return groups

    @staticmethod
    def _to_document_chunk(item: dict[str, Any]) -> DocumentChunk:
        """Convert a VectorStore result into a DocumentChunk."""

        metadata = item.get("metadata") or {}

        position = metadata.get(
            "chunk_index",
            metadata.get("parent_index"),
        )

        if not isinstance(position, int) or isinstance(position, bool):
            position = None

        return DocumentChunk(
            chunk_id=item["id"],
            document_id=metadata.get("document_id", item["id"]),
            content=item.get("content", ""),
            metadata=metadata,
            parent_chunk_id=metadata.get("parent_id"),
            position=position,
        )

    def _merge_evidence(
        self,
        evidence_by_chunk_id: dict[str, Evidence],
        incoming: Evidence,
    ) -> None:
        """Deduplicate evidence and keep its highest-priority type."""

        existing = evidence_by_chunk_id.get(incoming.chunk.chunk_id)

        if existing is None:
            evidence_by_chunk_id[incoming.chunk.chunk_id] = incoming
            return

        related_query_ids = list(existing.related_query_ids)

        for query_id in incoming.related_query_ids:
            if query_id not in related_query_ids:
                related_query_ids.append(query_id)

        existing_priority = self._TYPE_PRIORITY[existing.evidence_type]
        incoming_priority = self._TYPE_PRIORITY[incoming.evidence_type]

        if incoming_priority > existing_priority:
            evidence_by_chunk_id[incoming.chunk.chunk_id] = Evidence(
                evidence_id=incoming.evidence_id,
                chunk=incoming.chunk,
                evidence_type=incoming.evidence_type,
                source_chunk_id=incoming.source_chunk_id,
                related_query_ids=related_query_ids,
                relevance_score=incoming.relevance_score,
            )
            return

        existing.related_query_ids = related_query_ids

        if (
            incoming.relevance_score is not None
            and (
                existing.relevance_score is None
                or incoming.relevance_score > existing.relevance_score
            )
        ):
            existing.relevance_score = incoming.relevance_score
