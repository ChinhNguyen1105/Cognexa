from typing import Any


class ResultAggregator:
    """
    Merge retrieval results from multiple queries using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        self.rrf_k = rrf_k

    def aggregate(
        self,
        retrieval_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Merge and rank documents from multiple retrieval results.

        Each document is identified by its unique ID.

        Returns:
            Unique documents sorted by RRF score.
        """

        aggregated: dict[str, dict[str, Any]] = {}

        for query_index, documents in enumerate(retrieval_results):

            for rank, document in enumerate(documents, start=1):

                if not isinstance(document, dict):
                    raise TypeError(
                        "Each document must be a dictionary."
                    )

                document_id = document.get("id")

                if (
                    not isinstance(document_id, str)
                    or not document_id.strip()
                ):
                    raise ValueError(
                        "Every document must have a valid ID."
                    )

                if document_id not in aggregated:
                    aggregated[document_id] = {
                        **document,
                        "rrf_score": 0.0,
                        "retrieval_ranks": [],
                        "matched_queries": [],
                    }

                item = aggregated[document_id]

                # Calculate RRF score.
                item["rrf_score"] += 1 / (self.rrf_k + rank)

                # Track the document's rank in each query.
                item["retrieval_ranks"].append({
                    "query_index": query_index,
                    "rank": rank,
                })

                # Track which queries retrieved this document.
                item["matched_queries"].append(query_index)

                # Preserve the smallest distance for reference.
                current_distance = document.get("distance")
                previous_distance = item.get("distance")

                if (
                    isinstance(current_distance, (int, float))
                    and (
                        not isinstance(previous_distance, (int, float))
                        or current_distance < previous_distance
                    )
                ):
                    item["distance"] = current_distance

        return sorted(
            aggregated.values(),
            key=lambda document: document["rrf_score"],
            reverse=True,
        )


