
from cognexa.knowledge.retrieval.ranking.reranker import (
    CrossEncoderReranker,
)


reranker = CrossEncoderReranker()

questions = [
    "what does python widely use for",
    "what programming paradigm does python support",
    "what is python",
]

candidates = [
    "Python is a high-level programming language.",
    "Python supports object-oriented programming.",
    "Python is widely used in artificial intelligence and machine learning.",
]

for question in questions:
    print("\n" + "=" * 75)
    print(f"QUESTION: {question}")

    results = reranker.rerank(
        query=question,
        candidates=candidates,
    )

    for result in results:
        print(
            f"{result['score']:.4f} | "
            f"{result['sentence']}"
        )


