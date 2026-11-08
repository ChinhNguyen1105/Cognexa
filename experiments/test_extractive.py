from cognexa.knowledge.validation.answer_verifier import (
    AnswerVerifier,
)

context = """
Python is a high-level programming language.
Python supports object-oriented programming.
Python is widely used in artificial intelligence and machine learning.
"""

question = "what is python"

verifier = AnswerVerifier()

sentences = [
    sentence.strip()
    for sentence in context.split(".")
    if sentence.strip()
]

print(f"Question: {question}\n")

for sentence in sentences:
    claim = sentence + "."

    score = verifier._get_entailment_score(
        context=context,
        hypothesis=claim,
    )

    print(f"Sentence: {claim}")
    print(f"Score: {score:.4f}")