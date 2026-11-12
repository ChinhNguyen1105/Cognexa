
import json
import re

from cognexa.llm.generator import TextGenerator

from .query_analyzer import QueryAnalyzer


class QuestionDecomposer:
    """
    Decompose complex questions into smaller sub-questions.

    Uses structured LLM output with aspect labels.
    Keeps the original question and validates decomposition
    before returning retrieval queries.
    """

    REQUIRED_ASPECTS = {
        "comparison": {"comparison"},
        "cause_effect": {"causes", "consequences"},
        "multi_aspect": set(),
        "relationship": set(),
    }

    def __init__(
        self,
        generator: TextGenerator,
        analyzer: QueryAnalyzer | None = None,
    ):
        self.generator = generator
        self.analyzer = analyzer or QueryAnalyzer()

    def decompose(self, question: str) -> list[str]:
        """
        Backward-compatible API returning retrieval query strings.
        """
        return [
            item["question"]
            for item in self.decompose_detailed(question)
        ]

    def decompose_detailed(self, question: str) -> list[dict[str, str]]:
        """
        Return retrieval queries while preserving their aspect labels.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        question_type = self.analyzer.classify(question)

        if question_type == "simple":
            return [
                {
                    "aspect": "original",
                    "question": question,
                }
            ]

        messages = self._build_prompt(
            question=question,
            question_type=question_type,
        )

        print("\n" + "=" * 70)
        print("[DECOMPOSITION DEBUG]")
        print(f"Question type: {question_type}")
        print(f"Original question: {question}")

        response = self.generator.generate(
            messages,
            max_new_tokens=256,
        )

        print("\n[1] RAW LLM RESPONSE")
        print(response)

        try:
            structured_questions = self._parse_response(response)

            cleaned_items = self._validate_sub_questions(
                structured_questions
            )

            is_valid = self._covers_question_type(
                original_question=question,
                sub_questions=cleaned_items,
                question_type=question_type,
            )

        except (ValueError, TypeError) as error:
            print("\n[VALIDATION RESULT]")
            print("Accepted: False")
            print(f"Reason: {error}")
            print("Action: Fallback to original question.")
            print("=" * 70 + "\n")
            return [
                {
                    "aspect": "original",
                    "question": question,
                }
            ]

        print("\n[2] VALIDATED SUB-QUESTIONS")

        for index, item in enumerate(cleaned_items, start=1):
            print(
                f"{index}. [{item['aspect']}] "
                f"{item['question']}"
            )

        print("\n[3] VALIDATION RESULT")
        print(f"Accepted: {is_valid}")

        if not is_valid:
            print("Reason: Question-type coverage validation failed.")
            print("Action: Fallback to original question.")

        print("=" * 70 + "\n")

        if not is_valid:
            return [
                {
                    "aspect": "original",
                    "question": question,
                }
            ]

        return [
            {
                "aspect": "original",
                "question": question,
            },
            *cleaned_items,
        ]

    @staticmethod
    def _build_prompt(
        question: str,
        question_type: str,
    ) -> list[dict[str, str]]:
        system_prompt = """
You are a precise question decomposition system.

Your task is to decompose a complex question into
2 to 4 concise, independent sub-questions.

IMPORTANT:
- Generate at least 2 and at most 4 sub-questions.
- Never generate more than 4 sub-questions.
- Do not generate unnecessary questions just to reach the limit.
- Every sub-question must contribute to answering the original question.
- Cover all explicitly requested information needs.

GENERAL RULES:
1. Preserve the original meaning and intent.
2. Preserve all technical terms and named entities.
3. Each sub-question must address a distinct information need.
4. Avoid redundant or overlapping questions.
5. Do not introduce unrelated topics.
6. Every question must be grammatically complete.
7. Every question must end with a question mark.
8. Do not produce statements, sentence fragments, or headings.

QUESTION-TYPE RULES:

A. COMPARISON

The original question asks to compare two or more subjects.

Requirements:
- Identify and preserve all subjects being compared.
- Generate at least one direct comparison question.
- The direct comparison question MUST use aspect "comparison".
- The direct comparison question must explicitly mention
  all subjects being compared.
- If useful, generate additional questions about
  individual characteristics of the subjects.
- Do not replace comparison with unrelated definitions.

Example:
Original: Compare RAG and fine-tuning.

Output:
{
  "sub_questions": [
    {
      "aspect": "comparison",
      "question": "How do RAG and fine-tuning differ in their approaches to using external knowledge?"
    },
    {
      "aspect": "comparison",
      "question": "How do RAG and fine-tuning differ in their training and inference processes?"
    }
  ]
}

B. CAUSE AND EFFECT

The original question asks about causes and consequences.

Requirements:
- Generate at least one question about causes.
- Generate at least one question about consequences.
- Use EXACTLY "causes" as the aspect for cause-related questions.
- Use EXACTLY "consequences" as the aspect for effect-related questions.
- Preserve the direction of causality.
- Do not replace consequences with additional causes.
- Do not use "cause_effect" or "definition" as aspect labels
  for these required questions.

Example:
Original: What are the causes and consequences of overfitting?

Output:
{
  "sub_questions": [
    {
      "aspect": "causes",
      "question": "What factors cause overfitting in machine learning models?"
    },
    {
      "aspect": "consequences",
      "question": "How does overfitting affect a model's performance on unseen data?"
    }
  ]
}

C. MULTI-ASPECT

The original question contains multiple information needs.

Requirements:
- Identify every explicitly requested information need.
- Use concise aspect labels.
- For definition questions, use "definition".
- For working mechanism questions, use "process".
- For benefit questions, use "benefits".
- For limitation questions, use "limitations".
- Do not generate questions about aspects that are not relevant.
- Prioritize explicitly requested aspects over additional information.
- Use no more than 4 sub-questions.

Example:
Original: What is RAG and how does it work?

Output:
{
  "sub_questions": [
    {
      "aspect": "definition",
      "question": "What is Retrieval-Augmented Generation (RAG)?"
    },
    {
      "aspect": "process",
      "question": "How does the RAG process retrieve information and generate answers?"
    }
  ]
}

D. RELATIONSHIP

The original question asks about the relationship between entities.

Requirements:
- Preserve all entities mentioned in the original question.
- Generate questions that investigate their relationship.
- Use relevant dimensions such as similarities,
  differences, interaction, dependency, or shared characteristics.
- Each question must explicitly mention all entities
  whose relationship is being investigated.
- Do not focus exclusively on unrelated individual definitions.
- Use meaningful aspect labels.

Example:
Original: What is the relationship between cats and dogs?

Output:
{
  "sub_questions": [
    {
      "aspect": "similarities",
      "question": "What characteristics do cats and dogs share as domesticated animals?"
    },
    {
      "aspect": "differences",
      "question": "How do cats and dogs differ in their behavior and communication?"
    }
  ]
}

OUTPUT FORMAT:

Return ONLY a valid JSON object using this structure:

{
  "sub_questions": [
    {
      "aspect": "short_aspect_label",
      "question": "A complete sub-question?"
    }
  ]
}

STRICT OUTPUT CONSTRAINTS:
- The root must be a JSON object.
- The object must contain the "sub_questions" array.
- Every array item must contain string fields "aspect" and "question".
- Generate between 2 and 4 items.
- Use double quotes for JSON strings.
- Do not include trailing commas.
- Do not include Markdown code fences.
- Do not include explanations before or after the JSON.
"""

        user_prompt = (
            f"Question type: {question_type}\n\n"
            f"Original question:\n{question}\n\n"
            "Generate the required sub-questions. "
            "Follow the rules for this question type exactly."
        )

        return [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    @staticmethod
    def _parse_response(response: str) -> list[dict]:
        if not isinstance(response, str):
            raise ValueError("The LLM response must be a string.")

        response = response.strip()

        if not response:
            raise ValueError("The LLM returned an empty response.")

        if response.startswith("```"):
            lines = response.splitlines()

            if lines[-1].strip().startswith("```"):
                response = "\n".join(lines[1:-1]).strip()

        try:
            result = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError(
                "The LLM returned an invalid JSON response."
            ) from error

        if not isinstance(result, dict):
            raise ValueError(
                "The LLM response must be a JSON object."
            )

        if "sub_questions" not in result:
            raise ValueError(
                "The response must contain sub_questions."
            )

        sub_questions = result["sub_questions"]

        if not isinstance(sub_questions, list):
            raise ValueError(
                "sub_questions must be a list."
            )

        return sub_questions

    @staticmethod
    def _validate_sub_questions(
        sub_questions: list,
    ) -> list[dict[str, str]]:
        cleaned_items = []
        seen = set()

        for item in sub_questions:
            if not isinstance(item, dict):
                raise ValueError(
                    "Every sub-question must be an object."
                )

            if "aspect" not in item:
                raise ValueError(
                    "Every sub-question must contain aspect."
                )

            if "question" not in item:
                raise ValueError(
                    "Every sub-question must contain question."
                )

            aspect = item["aspect"]
            question = item["question"]

            if not isinstance(aspect, str):
                raise ValueError(
                    "Every aspect must be a string."
                )

            if not isinstance(question, str):
                raise ValueError(
                    "Every sub-question must be a string."
                )

            aspect = aspect.strip().casefold()
            question = question.strip()

            if not aspect or not question:
                continue

            normalized = question.casefold()

            if normalized in seen:
                continue

            cleaned_items.append({
                "aspect": aspect,
                "question": question,
            })
            seen.add(normalized)

        if len(cleaned_items) < 2:
            raise ValueError(
                "At least two valid sub-questions are required."
            )

        if len(cleaned_items) > 4:
            raise ValueError(
                "The LLM generated too many sub-questions."
            )

        return cleaned_items

    @classmethod
    def _covers_question_type(
        cls,
        original_question: str,
        sub_questions: list[dict[str, str]],
        question_type: str,
    ) -> bool:
        if question_type == "comparison":
            return cls._valid_comparison(
                original_question,
                sub_questions,
            )

        if question_type == "cause_effect":
            return cls._valid_cause_effect(sub_questions)

        if question_type == "multi_aspect":
            return cls._valid_multi_aspect(
                original_question,
                sub_questions,
            )

        if question_type == "relationship":
            return cls._valid_relationship(
                original_question,
                sub_questions,
            )

        return True

    @staticmethod
    def _valid_comparison(
        original_question: str,
        sub_questions: list[dict[str, str]],
    ) -> bool:
        original = original_question.casefold()
        combined = " ".join(
            item["question"].casefold()
            for item in sub_questions
        )

        match = re.search(
            r"\bcompare\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
            original,
        )

        if not match:
            return False

        first_subject = match.group(1).strip()
        second_subject = match.group(2).strip()

        if (
            first_subject not in combined
            or second_subject not in combined
        ):
            return False

        return any(
            item["aspect"] == "comparison"
            for item in sub_questions
        )

    @staticmethod
    def _valid_cause_effect(
        sub_questions: list[dict[str, str]],
    ) -> bool:
        aspects = {
            item["aspect"]
            for item in sub_questions
        }

        return {
            "causes",
            "consequences",
        }.issubset(aspects)

    @staticmethod
    def _valid_multi_aspect(
        original_question: str,
        sub_questions: list[dict[str, str]],
    ) -> bool:
        original = original_question.casefold()
        aspects = {
            item["aspect"]
            for item in sub_questions
        }

        required_aspects = set()

        if re.search(r"\bwhat\s+is\b", original):
            required_aspects.add("definition")

        if re.search(r"\bhow\b", original):
            required_aspects.add("process")

        if re.search(r"\bwhy\b", original):
            required_aspects.add("reasons")

        aliases = {
            "definition": {"definition", "concept", "meaning"},
            "process": {"process", "working", "mechanism"},
            "reasons": {"reasons", "benefits", "purpose"},
        }

        for required in required_aspects:
            if not aspects.intersection(aliases[required]):
                return False

        return True

    @staticmethod
    def _valid_relationship(
        original_question: str,
        sub_questions: list[dict[str, str]],
    ) -> bool:
        match = re.search(
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
            original_question.casefold(),
        )

        if not match:
            return False

        first_subject = match.group(1).strip()
        second_subject = match.group(2).strip()

        for item in sub_questions:
            sub_question = item["question"].casefold()

            if (
                first_subject in sub_question
                and second_subject in sub_question
            ):
                return True

        return False

    @staticmethod
    def _merge_queries(
        original_question: str,
        sub_questions: list[str],
    ) -> list[str]:
        queries = [original_question]
        seen = {original_question.casefold()}

        for question in sub_questions:
            normalized = question.strip().casefold()

            if normalized in seen:
                continue

            queries.append(question)
            seen.add(normalized)

        return queries


