from cognexa.knowledge.embedding.embedder import TextEmbedder
from cognexa.assistant.answer_extractor import AnswerExtractor

embedder = TextEmbedder()

sentences = [
    "Python is a high-level programming language.",
    "Python supports object-oriented programming.",
    "Python is widely used in artificial intelligence and machine learning.",
]

questions = [
    "what is python",
    "what programming paradigm does python support",
]

for question in questions:
    print(f"\nQuestion: {question}")

    question_embedding = embedder.embed_text(question)
    sentence_embeddings = embedder.embed_texts(sentences)

    for sentence, embedding in zip(sentences, sentence_embeddings):
        similarity = AnswerExtractor._cosine_similarity(
            question_embedding,
            embedding,
        )

        print(f"{similarity:.4f} | {sentence}")