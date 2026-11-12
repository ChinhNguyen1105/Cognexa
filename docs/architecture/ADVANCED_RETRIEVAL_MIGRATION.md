# Cognexa Advanced Retrieval - Migration Plan

## Important

This scaffold does not move existing modules. Move modules only after checking
their public classes/functions and all imports. This prevents a directory-only
refactor from silently breaking the working v0.6/v0.7 pipeline.

## Existing module -> proposed responsibility

| Existing module | Proposed destination | Migration note |
|---|---|---|
| `knowledge/query/query_analyzer.py` | `knowledge/query/intent.py` and/or `normalizer.py` | Split only after inspecting its actual responsibilities. |
| `knowledge/query/question_decomposer.py` | `knowledge/query/decomposer.py` | Keep decomposition conditional; do not decompose every query. |
| `knowledge/retrieval/retriever.py` | `knowledge/retrieval/candidates/dense.py` | Only if it currently implements dense/vector retrieval. |
| New module | `knowledge/retrieval/candidates/sparse.py` | Add BM25 candidate retrieval. |
| `knowledge/retrieval/result_aggregator.py` | `knowledge/retrieval/fusion/rrf.py` | Reuse if it already implements RRF; otherwise separate aggregation from fusion. |
| `knowledge/reranking/reranker.py` | `knowledge/retrieval/ranking/cross_encoder.py` | Preserve the existing Cross-Encoder implementation and interface. |
| New module | `knowledge/retrieval/expansion/parent_child.py` | Retrieve parent context for matched child chunks. |
| New module | `knowledge/retrieval/expansion/neighbor.py` | Expand to adjacent chunks when evidence needs surrounding context. |
| `knowledge/validation/context_validator.py` | `knowledge/retrieval/coverage/evaluator.py` | First inspect whether it validates relevance, context quality, or coverage. |
| `assistant/answerability_checker.py` | `knowledge/retrieval/coverage/sufficiency.py` | Move only if it checks evidence sufficiency rather than final answer wording. |
| New module | `knowledge/retrieval/coverage/missing_evidence.py` | Identify uncovered sub-questions / missing evidence. |
| New module | `knowledge/retrieval/pipeline.py` | Orchestrate the seven retrieval stages. |
| `knowledge/processing/chunker.py` | `knowledge/ingestion/chunking/` | Keep the current implementation until semantic/parent-child chunking is designed. |
| `knowledge/vector_store/store.py` | Keep in `knowledge/vector_store/` | Vector storage is an infrastructure component, not a retrieval stage. |
| `knowledge/knowledge_base.py` | Keep temporarily | Later make this a thin facade over ingestion, indexing and retrieval. |
| `assistant/answer_extractor.py` | Keep in `assistant/` | Answer extraction is downstream of retrieval; do not mix it into retrieval. |
| `assistant/answer_selector.py` | Keep in `assistant/` | Downstream answer selection; not part of candidate retrieval. |
| `knowledge/validation/answer_verifier.py` | Keep temporarily | This validates an answer, not necessarily retrieval evidence coverage. |

## Recommended implementation order

1. Add BM25 sparse retrieval and evaluate it independently.
2. Fuse Dense + BM25 candidates with RRF.
3. Keep Cross-Encoder reranking and compare before/after metrics.
4. Add conditional query decomposition for multi-part questions.
5. Add parent-child and neighbor evidence expansion.
6. Add sub-question coverage and missing-evidence detection.
7. Add targeted adaptive retrieval with a strict retry limit.
8. Improve ingestion parsing/chunking and re-index the corpus when needed.

## Required contracts between stages

- Every retrieved chunk should retain: `chunk_id`, `document_id`, `text`, `metadata`,
  `source`, and its retrieval/ranking scores where available.
- Candidate retrievers should return a common result structure.
- Fusion should combine ranked lists, not compare raw Dense and BM25 scores directly.
- Reranking should preserve chunk identity and metadata.
- Expansion must deduplicate chunks and preserve parent/child relationships.
- Coverage should report covered sub-questions, missing sub-questions, and evidence IDs.
- Adaptive retrieval must stop when coverage is sufficient, no useful new evidence is found,
  or the configured retry limit is reached.

## Evaluation gates

Run the existing test suite before and after each migration. Add separate tests for:
- Dense retrieval
- BM25 retrieval
- RRF fusion
- Cross-Encoder ordering
- Parent/neighbor expansion and deduplication
- Sub-question coverage
- Adaptive retrieval stopping conditions

Do not use answer-generation quality as a substitute for retrieval metrics.
