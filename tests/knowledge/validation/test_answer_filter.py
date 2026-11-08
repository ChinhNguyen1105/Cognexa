from cognexa.knowledge.validation.answer_verifier import (
    AnswerVerifier,
)


def main():
    verifier = AnswerVerifier()

    context = """
    Python is a high-level programming language.
    Python supports object-oriented programming.
    Python is widely used in artificial intelligence
    and machine learning.
    """

    answer = """
    Python is a high-level programming language.
    Python was created by Guido van Rossum.
    Python supports object-oriented programming.
    Python was first released in 1991.
    """

    supported_sentences = verifier.get_supported_sentences(
        context=context,
        answer=answer,
    )

    print("Supported sentences:")

    for sentence in supported_sentences:
        print(f"- {sentence}")


if __name__ == "__main__":
    main()
