from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Rerank candidate sentences based on their relevance
    to a given query using a Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        ),
        device: str | None = None,
        batch_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # Lazy loading: load the model only when needed.
        self.model = None

    def _load_model(self):
        """
        Load the Cross-Encoder model on first use.
        """
        if self.model is None:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
            )

    def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[dict]:
        """
        Rank candidate sentences by query relevance.

        Returns:
            A list of dictionaries containing:
                - index: original candidate position
                - sentence: candidate sentence
                - score: relevance score
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not candidates:
            return []

        if any(
            not isinstance(candidate, str)
            or not candidate.strip()
            for candidate in candidates
        ):
            raise ValueError(
                "Candidates must contain non-empty strings."
            )

        self._load_model()

        pairs = [
            (query, candidate)
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked_results = []

        for index, (candidate, score) in enumerate(
            zip(candidates, scores)
        ):
            ranked_results.append(
                {
                    "index": index,
                    "sentence": candidate,
                    "score": float(score),
                }
            )

        ranked_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked_results


