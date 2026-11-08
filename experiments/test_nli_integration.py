from cognexa.knowledge.validation.answer_verifier import (
    AnswerVerifier,
)


def main():
    verifier = AnswerVerifier(
        min_entailment_score=0.70,
    )

    context = """
    Python is a high-level programming language.
    Python supports object-oriented programming.
    Python is widely used in artificial intelligence
    and machine learning.
    """

    test_cases = [
        (
            "Supported",
            "Python is a high-level programming language.",
        ),
        (
            "Unsupported",
            "Python was created by Guido van Rossum.",
        ),
        (
            "Unsupported",
            "Python was first released in 1991.",
        ),
        (
            "Contradictory",
            "Python does not support object-oriented programming.",
        ),
    ]

    for name, answer in test_cases:
        score = verifier._get_entailment_score(
            context=context,
            hypothesis=answer,
        )

        result = verifier.is_supported(
            context=context,
            answer=answer,
        )

        print("=" * 60)
        print(f"Test: {name}")
        print(f"Answer: {answer}")
        print(f"Entailment score: {score:.4f}")
        print(f"Supported: {result}")


if __name__ == "__main__":
    main()
