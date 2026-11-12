# Retrieval tests

Keep retrieval tests separate by responsibility:

- `unit/`: deterministic tests for individual retrieval components.
- `integration/`: end-to-end retrieval pipeline tests using a small fixed corpus.
- `../evaluation/`: golden-dataset evaluation and retrieval metrics.

Prefer small deterministic fixtures for unit tests. Keep model-backed evaluation
separate so ordinary test runs do not require downloading models.
