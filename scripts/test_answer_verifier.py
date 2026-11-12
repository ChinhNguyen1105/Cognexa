from cognexa.verification import AnswerVerifier


def main():
    print("=" * 60)
    print("COGNEXA - ANSWER VERIFIER INTEGRATION TEST")
    print("=" * 60)

    verifier = AnswerVerifier()

    test_cases = [
        {
            "name": "Supported",
            "answer": "Python is a programming language.",
            "evidence": [
                "Python is a programming language.",
            ],
            "expected": "supported",
        },
        {
            "name": "Contradicted",
            "answer": "Python is a compiled language.",
            "evidence": [
                "Python is generally described as an interpreted language.",
            ],
            "expected": "contradicted",
        },
        {
            "name": "Unsupported",
            "answer": "Python is used to build web applications.",
            "evidence": [
                "The weather is sunny today.",
            ],
            "expected": "unsupported",
        },
    ]

    for case in test_cases:
        result = verifier.verify(
            answer=case["answer"],
            evidence=case["evidence"],
        )

        actual = result.status.value
        passed = actual == case["expected"]

        print(f"\n[{ 'PASS' if passed else 'FAIL' }] {case['name']}")
        print(f"Expected: {case['expected']}")
        print(f"Actual:   {actual}")

        for sentence in result.sentences:
            print(f"Sentence: {sentence.sentence}")
            print(f"Status:   {sentence.status.value}")
            print(f"Reason:   {sentence.reason}")
            print(f"Scores:   E={sentence.entailment_score}, "
                  f"C={sentence.contradiction_score}, "
                  f"N={sentence.neutral_score}")

        if not passed:
            raise AssertionError(
                f"{case['name']} test failed."
            )

    print("\nAll AnswerVerifier integration tests passed.")


if __name__ == "__main__":
    main()
