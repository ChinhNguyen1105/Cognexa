from cognexa.verification.nli_model import NLIModel


def main():
    model = NLIModel()

    premises = [
        "Python is a programming language.",
        "Python is generally described as an interpreted language.",
        "Python was first released in 1991.",
    ]

    hypotheses = [
        "Python is a programming language.",
        "Python is a compiled language.",
        "Python was first released in 1995.",
    ]

    results = model.predict_batch(premises, hypotheses)

    for index, result in enumerate(results, start=1):
        total = (
            result.entailment
            + result.contradiction
            + result.neutral
        )

        print(f"\nPair {index}")
        print(f"Entailment:    {result.entailment}")
        print(f"Contradiction: {result.contradiction}")
        print(f"Neutral:       {result.neutral}")
        print(f"Total:         {total}")
        print(f"Type:          {type(result.entailment)}")


if __name__ == "__main__":
    main()
