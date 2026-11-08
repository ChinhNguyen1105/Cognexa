class ContextValidator:
    """
    Validate whether retrieved documents are semantically
    relevant to the user's question.
    """

    def __init__(self, max_distance: float = 0.55):
        if max_distance < 0:
            raise ValueError(
                "max_distance must be greater than or equal to 0."
            )

        self.max_distance = max_distance

    def is_relevant(
        self,
        documents: list[dict],
    ) -> bool:
        if not documents:
            print("[DEBUG VALIDATOR] No documents received.")
            return False

        print("\n[DEBUG VALIDATOR]")
        print(f"Max distance: {self.max_distance}")

        for index, document in enumerate(documents, start=1):
            distance = document.get("distance")

            print(
                f"Document {index} | "
                f"Distance: {distance} | "
                f"Threshold: {self.max_distance}"
            )

            if distance is None:
                print(" -> Skipped: distance is None")
                continue

            if distance <= self.max_distance:
                print(" -> Relevant")
                return True

            print(" -> Not relevant")

        print("[DEBUG VALIDATOR] All documents rejected.")

        return False