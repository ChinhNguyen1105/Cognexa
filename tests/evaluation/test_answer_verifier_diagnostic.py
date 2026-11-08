import pytest


def test_answer_verifier_diagnostic(rag_environment):
    verifier = rag_environment.answer_verifier

    test_cases = [
        {
            "name": "OVF-004 - Data leakage",
            "context": (
                "Data leakage, noisy labels, insufficient "
                "regularization, and poorly designed validation "
                "processes can contribute to overfitting."
            ),
            "hypothesis": (
                "Data leakage can contribute to overfitting."
            ),
        },
        {
            "name": "OVF-004 - Noisy labels",
            "context": (
                "Data leakage, noisy labels, insufficient "
                "regularization, and poorly designed validation "
                "processes can contribute to overfitting."
            ),
            "hypothesis": (
                "Noisy labels can contribute to overfitting."
            ),
        },
        {
            "name": "OVF-004 - Insufficient regularization",
            "context": (
                "Data leakage, noisy labels, insufficient "
                "regularization, and poorly designed validation "
                "processes can contribute to overfitting."
            ),
            "hypothesis": (
                "Insufficient regularization can contribute "
                "to overfitting."
            ),
        },
        {
            "name": "OVF-006 - L1 regularization",
            "context": (
                "L1 regularization, L2 regularization, dropout, "
                "and early stopping are techniques used to "
                "reduce overfitting."
            ),
            "hypothesis": (
                "L1 regularization can help reduce overfitting."
            ),
        },
        {
            "name": "OVF-007 - Overfitting",
            "context": (
                "An overfitted model performs well on training "
                "data but poorly on unseen data."
            ),
            "hypothesis": (
                "An overfitted model performs well on training "
                "data but poorly on unseen data."
            ),
        },
    ]

    for case in test_cases:
        result = verifier.verify_sentence(
            context=case["context"],
            hypothesis=case["hypothesis"],
        )

        print("\n" + "=" * 70)
        print(f"TEST: {case['name']}")
        print(f"PREMISE: {case['context']}")
        print(f"HYPOTHESIS: {case['hypothesis']}")
        print(f"STATUS: {result['status']}")
        print(f"METHOD: {result['method']}")
        print(f"SCORE: {result['score']}")
        print(
            f"CONTRADICTION: "
            f"{result.get('contradiction_score')}"
        )

    assert len(test_cases) == 5