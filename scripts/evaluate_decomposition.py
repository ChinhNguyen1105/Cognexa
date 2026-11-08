from cognexa.cli.assistant.commands import create_generator
from cognexa.knowledge.query.query_analyzer import QueryAnalyzer
from cognexa.knowledge.query.question_decomposer import QuestionDecomposer


EVALUATION_DATASET = [
    {
        "category": "Simple",
        "question": "What is RAG?",
        "expected_decomposition": False,
        "expected_aspects": ["Definition of RAG"],
    },
    {
        "category": "Simple",
        "question": "Explain how embeddings work.",
        "expected_decomposition": False,
        "expected_aspects": ["How embeddings work"],
    },
    {
        "category": "Multi-aspect",
        "question": "What is RAG and how does it work?",
        "expected_decomposition": True,
        "expected_aspects": [
            "Definition of RAG",
            "How RAG works",
        ],
    },
    {
        "category": "Comparison",
        "question": "Compare RAG and fine-tuning.",
        "expected_decomposition": True,
        "expected_aspects": [
            "RAG characteristics",
            "Fine-tuning characteristics",
            "Comparison criteria",
        ],
    },
    {
        "category": "Cause and effect",
        "question": "What are the causes and consequences of overfitting?",
        "expected_decomposition": True,
        "expected_aspects": [
            "Causes of overfitting",
            "Consequences of overfitting",
        ],
    },
    {
        "category": "Relationship",
        "question": "What is the relationship between cats and dogs?",
        "expected_decomposition": False,
        "expected_aspects": [
            "Relationship between cats and dogs",
        ],
    },
]


def evaluate():
    analyzer = QueryAnalyzer()

    print("=" * 70)
    print("COGNEXA - QUERY DECOMPOSITION EVALUATION")
    print("=" * 70)

    print("\nLoading LLM...")

    generator = create_generator()

    decomposer = QuestionDecomposer(
        generator=generator,
        analyzer=analyzer,
    )

    print("LLM loaded successfully.\n")

    total = len(EVALUATION_DATASET)
    analyzer_correct = 0

    for index, sample in enumerate(EVALUATION_DATASET, start=1):
        question = sample["question"]
        expected = sample["expected_decomposition"]

        print("-" * 70)
        print(f"Test case: {index}/{total}")
        print(f"Category: {sample['category']}")
        print(f"Question: {question}")

        actual = analyzer.is_complex(question)

        if actual == expected:
            analyzer_correct += 1
            analyzer_status = "PASS"
        else:
            analyzer_status = "FAIL"

        print(f"\nExpected decomposition: {expected}")
        print(f"Actual decomposition:   {actual}")
        print(f"Analyzer result:        {analyzer_status}")

        try:
            sub_questions = decomposer.decompose(question)

            print("\nGenerated sub-questions:")

            for sub_index, sub_question in enumerate(
                sub_questions,
                start=1,
            ):
                print(f"  {sub_index}. {sub_question}")

            print("\nExpected aspects to cover:")

            for aspect in sample["expected_aspects"]:
                print(f"  - {aspect}")

            if actual:
                print("\nDecomposition count:", len(sub_questions))
            else:
                print("\nOriginal question preserved:", (
                    sub_questions == [question]
                ))

        except Exception as error:
            print("\nDecomposition error:")
            print(f"{type(error).__name__}: {error}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total test cases: {total}")
    print(f"Analyzer correct: {analyzer_correct}/{total}")

    accuracy = analyzer_correct / total * 100

    print(f"Analyzer accuracy: {accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
