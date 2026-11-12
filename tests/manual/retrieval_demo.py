from cognexa.cli.assistant.commands import create_knowledge_base


def main():
    knowledge_base = create_knowledge_base()

    queries = [
    "what is python",
    "what programming paradigm does python support",
    "what is python used for",
    "who created python",
    "when was python released",
    "what is the capital of Japan",
]

    for query in queries:
        print("=" * 60)
        print(f"Query: {query}")

        results = knowledge_base.search(
            query=query,
            top_k=3,
        )

        if not results:
            print("No documents retrieved.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Content: {result['content']}")


if __name__ == "__main__":
    main()

