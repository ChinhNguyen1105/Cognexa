from cognexa.knowledge.validation.answer_verifier import (
    AnswerVerifier,
)

context = """
Python is a high-level programming language.
Python supports object-oriented programming.
Python is widely used in artificial intelligence and machine learning.
"""

claims = [
    "Python is a high-level programming language.",
    "Python is a high-level programming language designed for easy development of scripts.",
    "Python supports object-oriented programming.",
    "Python is used in web development.",
]

verifier = AnswerVerifier(min_entailment_score=0.0)

for claim in claims:
    score = verifier._get_entailment_score(
        context=context,
        hypothesis=claim,
    )

    print(f"\nClaim: {claim}")
    print(f"Entailment score: {score:.4f}")