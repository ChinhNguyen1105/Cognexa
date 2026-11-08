from statistics import mean

from cognexa.cli.assistant.commands import (
    create_embedder,
    create_generator,
    create_vector_store,
)

from cognexa.knowledge.retrieval.retriever import Retriever
from cognexa.knowledge.query.question_decomposer import QuestionDecomposer


TOP_K = 3
MAX_DISTANCE = 0.28

EVALUATION_DATASET = [
    {
        "category": "Simple",
        "question": "What is RAG?",
    },
    {
        "category": "Multi-aspect",
        "question": "What is RAG and how does it work?",
    },
    {
        "category": "Comparison",
        "question": "Compare RAG and fine-tuning.",
    },
    {
        "category": "Cause and effect",
        "question": "What are the causes and consequences of overfitting?",
    },
    {
        "category": "Relationship",
        "question": "What is the relationship between cats and dogs?",
    },
]


def retrieve_documents(retriever, queries, top_k):
    """
    Retrieve documents for multiple queries and deduplicate by ID.

    If a document appears in multiple query results, keep
    the occurrence with the smallest distance.
    """
    documents_by_id = {}

    for query in queries:
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        for document in results:
            document_id = document["id"]

            existing = documents_by_id.get(document_id)

            if (
                existing is None
                or document["distance"] < existing["distance"]
            ):
                documents_by_id[document_id] = document

    unique_documents = list(documents_by_id.values())

    unique_documents.sort(
        key=lambda document: document["distance"]
    )

    return unique_documents


def calculate_metrics(documents):
    distances = [
        document["distance"]
        for document in documents
        if document.get("distance") is not None
    ]

    relevant_count = sum(
        distance <= MAX_DISTANCE
        for distance in distances
    )

    return {
        "count": len(documents),
        "relevant_count": relevant_count,
        "average_distance": mean(distances) if distances else None,
    }


def print_metrics(label, metrics):
    average_distance = metrics["average_distance"]

    distance_text = (
        f"{average_distance:.4f}"
        if average_distance is not None
        else "N/A"
    )

    print(f"\n{label}")
    print(f"  Unique documents: {metrics['count']}")
    print(f"  Relevant by distance threshold: {metrics['relevant_count']}")
    print(f"  Average distance: {distance_text}")


def print_documents(label, documents):
    print("\n" + "-" * 75)
    print(label)
    print("-" * 75)

    if not documents:
        print("No documents retrieved.")
        return

    for index, document in enumerate(documents, start=1):
        metadata = document.get("metadata") or {}

        file_name = metadata.get("file_name", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        distance = document.get("distance")

        print(f"\n[{index}] {file_name} | Chunk {chunk_index}")
        print(f"Distance: {distance:.4f}" if distance is not None else "Distance: N/A")
        print(f"Content: {document.get('content', '')}")


def evaluate():
    print("=" * 75)
    print("COGNEXA - RETRIEVAL QUALITY EVALUATION")
    print("=" * 75)

    print("\nInitializing components...")

    embedder = create_embedder()
    vector_store = create_vector_store()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    generator = create_generator()

    decomposer = QuestionDecomposer(
        generator=generator,
    )

    print("Initialization completed.")
    print(f"Knowledge base documents: {vector_store.count()}")

    if vector_store.count() == 0:
        print("Knowledge base is empty. Add documents before evaluation.")
        return

    summaries = []

    for index, sample in enumerate(EVALUATION_DATASET, start=1):
        question = sample["question"]

        print("\n" + "=" * 75)
        print(f"Test case: {index}/{len(EVALUATION_DATASET)}")
        print(f"Category: {sample['category']}")
        print(f"Question: {question}")

        try:
            sub_questions = decomposer.decompose(question)

            print("\nGenerated queries:")

            for query_index, query in enumerate(sub_questions, start=1):
                print(f"  {query_index}. {query}")

            baseline_documents = retrieve_documents(
                retriever,
                [question],
                TOP_K,
            )

            matched_budget = TOP_K * len(sub_questions)

            budget_documents = retrieve_documents(
                retriever,
                [question],
                matched_budget,
            )

            adaptive_documents = retrieve_documents(
                retriever,
                sub_questions,
                TOP_K,
            )

            baseline_metrics = calculate_metrics(baseline_documents)
            budget_metrics = calculate_metrics(budget_documents)
            adaptive_metrics = calculate_metrics(adaptive_documents)

            print_metrics("BASELINE - TOP 3", baseline_metrics)

            print_metrics(
                f"BASELINE - BUDGET MATCHED (TOP {matched_budget})",
                budget_metrics,
            )

            print_metrics(
                "ADAPTIVE DECOMPOSITION",
                adaptive_metrics,
            )

            baseline_ids = {
                document["id"]
                for document in baseline_documents
            }

            adaptive_ids = {
                document["id"]
                for document in adaptive_documents
            }

            additional_documents = [
                document
                for document in adaptive_documents
                if document["id"] not in baseline_ids
            ]

            print(
                "\nAdditional unique documents from decomposition:",
                len(additional_documents),
            )

            print_documents(
                "BASELINE RETRIEVED DOCUMENTS",
                baseline_documents,
            )

            print_documents(
                "ADAPTIVE RETRIEVED DOCUMENTS",
                adaptive_documents,
            )

            print_documents(
                "ADDITIONAL DOCUMENTS FROM ADAPTIVE",
                additional_documents,
            )

            summaries.append({
                "question": question,
                "baseline": baseline_metrics,
                "budget": budget_metrics,
                "adaptive": adaptive_metrics,
                "additional_documents": len(additional_documents),
            })

        except Exception as error:
            print(f"\nEvaluation error: {type(error).__name__}: {error}")

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    if not summaries:
        print("No successful test cases.")
        return

    print(f"Successful test cases: {len(summaries)}")

    print(
        "Average unique documents - Baseline Top 3:",
        f"{mean(item['baseline']['count'] for item in summaries):.2f}",
    )

    print(
        "Average unique documents - Budget matched:",
        f"{mean(item['budget']['count'] for item in summaries):.2f}",
    )

    print(
        "Average unique documents - Adaptive:",
        f"{mean(item['adaptive']['count'] for item in summaries):.2f}",
    )

    print(
        "Average additional documents:",
        f"{mean(item['additional_documents'] for item in summaries):.2f}",
    )

    print("=" * 75)


if __name__ == "__main__":
    evaluate()
