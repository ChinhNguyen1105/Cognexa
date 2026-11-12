# Cognexa test structure

The test tree mirrors `src/cognexa` wherever a test belongs to one source component.

## Test responsibilities

- `assistant/`: assistant behavior, conversation, prompt and answer handling.
- `cli/`: CLI entry points and commands.
- `knowledge/`: tests for individual knowledge components, organized like `src/cognexa/knowledge`.
- `integration/`: tests spanning multiple modules, such as hybrid retrieval and evidence expansion.
- `evaluation/`: end-to-end retrieval quality and benchmark datasets.
- `llm/`: configuration, tokenizer, model and generator tests.

## Advanced Retrieval mapping

| Source module | Unit-test location |
|---|---|
| `knowledge/query/` | `knowledge/query/` |
| `knowledge/retrieval/candidates/` | `knowledge/retrieval/candidates/` |
| `knowledge/retrieval/fusion/` | `knowledge/retrieval/fusion/` |
| `knowledge/retrieval/ranking/` | `knowledge/retrieval/ranking/` |
| `knowledge/retrieval/expansion/` | `knowledge/retrieval/expansion/` |
| `knowledge/retrieval/coverage/` | `knowledge/retrieval/coverage/` |
| Cross-module retrieval flow | `integration/` |
| Retrieval quality and benchmark metrics | `evaluation/` |

Do not add tests for planned modules until their implementation exists.
Keep unit tests deterministic. Use integration tests to verify handoffs between modules.
Use evaluation datasets and metrics to demonstrate actual relevance/completeness improvements.

## After migration

Run the full suite from the project root:

```powershell
python -m pytest -q
```

If a moved test uses relative imports or references files using paths relative to its old directory,
update those paths deliberately and rerun the affected test before running the full suite.
