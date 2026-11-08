
from cognexa.cli.assistant.commands import (
    create_knowledge_base,
    create_embedder,
)
from cognexa.assistant.answer_extractor import AnswerExtractor


questions = [
    "when people use python",
    "What programming paradigm does Python support?",
    "In which domain does python concentrate in",
    "What can python use",
    "What are Python's characteristics and applications?",
    "what does python widely use for",
]

sentences = [
    "Python is a high-level programming language.",
    "Python supports object-oriented programming.",
    "Python is widely used in artificial intelligence and machine learning.",
]

embedder = create_embedder()
knowledge_base = create_knowledge_base(embedder=embedder)

for question in questions:
    print("\n" + "=" * 75)
    print(f"QUESTION: {question}")

    documents = knowledge_base.search(
        query=question,
        top_k=3,
    )

    print("\n[1] RETRIEVAL")

    for document in documents:
        print(
            f"Distance: {document.get('distance', 'N/A'):.4f}"
            if document.get("distance") is not None
            else "Distance: N/A"
        )

        print(f"Content: {document.get('content', '')}")

    print("\n[2] SENTENCE SIMILARITY")

    question_embedding = embedder.embed_text(question)
    sentence_embeddings = embedder.embed_texts(sentences)

    scores = []

    for sentence, embedding in zip(
        sentences,
        sentence_embeddings,
    ):
        similarity = AnswerExtractor._cosine_similarity(
            question_embedding,
            embedding,
        )

        scores.append((similarity, sentence))

    scores.sort(reverse=True)

    for similarity, sentence in scores:
        print(f"{similarity:.4f} | {sentence}")
