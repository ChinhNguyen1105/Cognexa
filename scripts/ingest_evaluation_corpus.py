from pathlib import Path

from cognexa.cli.assistant.commands import create_knowledge_base


def main():
    documents_dir = Path("documents/evaluation")

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Document directory not found: {documents_dir}"
        )

    knowledge_base = create_knowledge_base()

    vector_store = knowledge_base.ingestion_pipeline.vector_store
    collection = vector_store.collection

    # Collect sources that have already been ingested.
    existing = collection.get(
        include=["metadatas"]
    )

    existing_sources = {
        metadata.get("source")
        for metadata in existing["metadatas"]
        if metadata and metadata.get("source")
    }

    files = sorted(documents_dir.glob("*.txt"))

    if not files:
        print("No .txt files found.")
        return

    print("Starting evaluation corpus ingestion...")
    print(f"Files found: {len(files)}")
    print()

    for file_path in files:
        source = str(file_path.resolve())

        if source in existing_sources:
            print(f"[SKIPPED] {file_path.name} (already ingested)")
            continue

        result = knowledge_base.add_document(file_path)

        print(f"[INGESTED] {file_path.name}")
        print(f"  Chunks: {result['chunk_count']}")

    # Verify the final collection.
    stored = collection.get(
        include=["metadatas"]
    )

    source_counts = {}

    for metadata in stored["metadatas"]:
        if not metadata:
            continue

        source = metadata.get("file_name", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print()
    print("=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total documents (chunks): {vector_store.count()}")
    print()

    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count} chunks")


if __name__ == "__main__":
    main()
